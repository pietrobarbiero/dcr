"""
General utils for training, evaluation and data loading

Heavily adapted from https://github.com/xmed-lab/ECBM/blob/main/data/awa2.py
Credit goes to Xinyue Xu, Yi Qin, Lu Mi, Hao Wang, and Xiaomeng Li
and the code accompanying their paper "Energy-Based Concept Bottleneck Models:
Unifying Prediction, Concept Intervention, and Probabilistic Interpretations"
"""
import numpy as np
import os
import sklearn
import torch
import torchvision.transforms as transforms

from functools import reduce
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, Subset, DataLoader

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 50

# CAN BE OVERWRITTEN WITH AN ENV VARIABLE DATASET_DIR
DATASET_DIR = os.environ.get("DATASET_DIR", 'data/AwA2/')


#########################################################
## CONCEPT INFORMATION REGARDING AwA2
#########################################################

SElECTED_CONCEPTS = list(range(1, 51))

CLASS_NAMES = [
    'antelope',
    'grizzly+bear',
    'killer+whale',
    'beaver',
    'dalmatian',
    'persian+cat',
    'horse',
    'german+shepherd',
    'blue+whale',
    'siamese+cat',
    'skunk',
    'mole',
    'tiger',
    'hippopotamus',
    'leopard',
    'moose',
    'spider+monkey',
    'humpback+whale',
    'elephant',
    'gorilla',
    'ox',
    'fox',
    'sheep',
    'seal',
    'chimpanzee',
    'hamster',
    'squirrel',
    'rhinoceros',
    'rabbit',
    'bat',
    'giraffe',
    'wolf',
    'chihuahua',
    'rat',
    'weasel',
    'otter',
    'buffalo',
    'zebra',
    'giant+panda',
    'deer',
    'bobcat',
    'pig',
    'lion',
    'mouse',
    'polar+bear',
    'collie',
    'walrus',
    'raccoon',
    'cow',
    'dolphin',
]

CONCEPT_SEMANTICS = [
    'black',
    'white',
    'blue',
    'brown',
    'gray',
    'orange',
    'red',
    'yellow',
    'patches',
    'spots',
    'stripes',
    'furry',
    'hairless',
    'toughskin',
    'big',
    'small',
    'bulbous',
    'lean',
    'flippers',
    'hands',
    'hooves',
    'pads',
    'paws',
    'longleg',
    'longneck',
    'tail',
    'chewteeth',
    'meatteeth',
    'buckteeth',
    'strainteeth',
    'horns',
    'claws',
    'tusks',
    'smelly',
    'flys',
    'hops',
    'swims',
    'tunnels',
    'walks',
    'fast',
    'slow',
    'strong',
    'weak',
    'muscle',
    'bipedal',
    'quadrapedal',
    'active',
    'inactive',
    'nocturnal',
    'hibernate',
    'agility',
    'fish',
    'meat',
    'plankton',
    'vegetation',
    'insects',
    'forager',
    'grazer',
    'hunter',
    'scavenger',
    'skimmer',
    'stalker',
    'newworld',
    'oldworld',
    'arctic',
    'coastal',
    'desert',
    'bush',
    'plains',
    'forest',
    'fields',
    'jungle',
    'mountains',
    'ocean',
    'ground',
    'water',
    'tree',
    'cave',
    'fierce',
    'timid',
    'smart',
    'group',
    'solitary',
    'nestspot',
    'domestic',
]





class AwA2Dataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the AwA2 dataset
    """

    def __init__(
        self,
        root_dir,
        augment_data=False,
        split='train',
        image_size=224,
        concept_transform=None,
        seed=42,
    ):
        self.root_dir = root_dir
        self.augment_data = augment_data
        self.split = split
        self.concept_transform = concept_transform

        if not os.path.exists(self.root_dir):
            raise ValueError(
                f'{self.root_dir} does not exist yet. Please download the '
                f'dataset first.'
            )


        if split == 'train':
            self.transform = get_transform_awa2(
                train=True,
                augment_data=augment_data,
                image_size=image_size,
            )
        else:
            self.transform = get_transform_awa2(
                train=False,
                augment_data=augment_data,
                image_size=image_size,
            )


        self.predicate_binary_mat = np.array(np.genfromtxt(
            os.path.join(root_dir, 'predicate-matrix-binary.txt'),
            dtype='int',
        ))
        self.class_to_index = dict()
        # Build dictionary of indices to classes
        with open(f"{root_dir}/classes.txt") as f:
            for line in f:
                class_name = line.split('\t')[1].strip()
                self.class_to_index[class_name] = len(self.class_to_index)

        for split_attempt in ['train', 'val', 'test']:
            split_file = os.path.join(
                self.root_dir,
                f'{split_attempt}_split.npz',
            )
            if not os.path.exists(split_file):
                print(
                    f"Split files for AWA2 could not be found. Generating new "
                    f"train, validation, and test splits with seed {seed}."
                )
                self._generate_splits(seed=seed)
                break

        # And now we can simply load the actual paths and classes to be used
        # for each split :)
        split_file = os.path.join(
            self.root_dir,
            f'{split}_split.npz',
        )
        split_info = np.load(split_file)
        self.img_paths = split_info['paths']
        self.img_labels = split_info['labels']
        print(f"{split.upper()} AWA2 dataset has:", len(self), f"samples")

    def _generate_splits(self, seed, train_size=0.6, val_size=0.2):
        # First find all samples and generate a list of their paths
        image_paths = []
        image_classes = []
        img_dir = os.path.join(self.root_dir, 'JPEGImages')
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith('.jpg'):
                    image_paths.append(os.path.abspath(os.path.join(root, file)))
                    parent_dir = os.path.basename(
                        os.path.dirname(image_paths[-1])
                    )
                    image_classes.append(self.class_to_index[parent_dir])

        np.random.seed(seed)
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)

        train_end = int(train_size * len(image_paths))
        val_end = train_end + int(val_size * len(image_paths))

        # Now time to generate our split matrices and saving them
        image_paths = np.array(image_paths)
        image_classes = np.array(image_classes)

        train_indices = indices[:train_end]
        train_paths = image_paths[train_indices]
        train_classes = image_classes[train_indices]
        np.savez(
            os.path.join(self.root_dir, 'train_split.npz'),
            paths=train_paths,
            labels=train_classes,
        )

        val_indices = indices[train_end:val_end]
        val_paths = image_paths[val_indices]
        val_classes = image_classes[val_indices]
        np.savez(
            os.path.join(self.root_dir, 'val_split.npz'),
            paths=val_paths,
            labels=val_classes,
        )

        test_indices = indices[val_end:]
        test_paths = image_paths[test_indices]
        test_classes = image_classes[test_indices]
        np.savez(
            os.path.join(self.root_dir, 'test_split.npz'),
            paths=test_paths,
            labels=test_classes,
        )


    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if img.getbands()[0] == 'L':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_idx = self.img_labels[index]
        concepts = self.predicate_binary_mat[label_idx,:]
        return img, label_idx, torch.FloatTensor(concepts)


    def __len__(self):
        return len(self.img_paths)


def get_transform_awa2(train, augment_data, image_size=224):
    """Helper function to get the appropiate transformation for the awa2
    data loader.

    Args:
        train (bool): Whether or not this transform is for the training fold
            of the awa2 dataset or not.
        augment_data (bool): Whether or not we want to perform standard
            augmentations (crops and flips) used for the CUB dataset.
        image_size (int, optional): Size of the width and height of each
            of the generated images. Defaults to 224.

    Returns:
        torchvision.Transform: a valid torchvision transform to be applied to
            each image of the awa2 dataset being loaded.
    """
    scale = 256.0/224.0
    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((
                int(image_size*scale),
                int(image_size*scale),
            )),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def load_data(
    split,
    batch_size,
    image_size=299,
    root_dir='data/awa2/',
    num_workers=1,
    dataset_transform=lambda x: x,
    dataset_size=None,
    augment_data=False,
    concept_transform=None,
):
    """Generates a Dataloader for the awa2 dataset.

    Args:
        split (str): Split of data to use for this loader. Must be one of
            'train', 'val', or 'test'.
        batch_size (int): Batch size used when loading data into the model.
        image_size (int, optional): Size of width and height of Waterbird
            images. Defaults to 299.
        root_dir (str, optional): Valid path to where the awa2 data
            is stored. Defaults to 'data/awa2/'.
        num_workers (int, optional): Number of data loader workers to use for
            the Data loader. Defaults to 1.
        dataset_transform (Func[torch.Tensor -> torch.Tensor], optional):
            If provided this corresponds to a transformation to be applied to
            each torch.Dataset object to be generated. Defaults to identify
            transform.
        dataset_size (int or float, optional): size of the output dataset.
            If an integer, then it is assumed to be the number of samples. If
            a float, then it should be between 0 and 1 and it represents the
            fraction of samples to select from the split used for this
            loader. Defaults to None which represents the whole dataset.
        augment_data (bool, optional): Whether or not we will augment the images
            in the resulting dataset. Defaults to False.
        class_dtype (builtin.type or np.type, optional): type to use for the
            labels. Defaults to float (i.e., binary label).

    Returns:
        torch.Dataloader: the corresponding data loader for the requested
            awa2 split.
    """
    dataset = AwA2Dataset(
        split=split,
        root_dir=root_dir,
        augment_data=augment_data,
        image_size=image_size,
        concept_transform=concept_transform,
    )
    if dataset_size is not None:
        # Then we will subsample this training set so that after splitting
        # into a training, test, and validation set we end up with
        # `dataset_size` samples in the training dataset
        train_idxs = np.random.permutation(len(dataset))
        if dataset_size > 0 and dataset_size < 1:
            # Then this is a fraction (function of the total size of the set!)
            dataset_size = int(
                np.ceil(len(dataset) * dataset_size)
            )
        train_idxs = train_idxs[:int(np.ceil(dataset_size))]
        dataset = Subset(
            dataset,
            train_idxs,
        )
    if split == 'train':
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    loader = DataLoader(
        dataset_transform(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return loader


########################
## Data Module
########################

def factors(n):
    return set(reduce(
        list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)
    ))

def get_num_labels(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: the number of class labels used in the awa2 dataset.
    """
    return N_CLASSES

def get_num_attributes(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: the number of attributes in the awa2 dataset.
    """
    return len(CONCEPT_SEMANTICS)

def generate_data(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
    dataset_transform=lambda x: x,
    training_transform=None,
):
    if root_dir is None:
        root_dir = DATASET_DIR
    seed_everything(seed)

    training_transform = (
        training_transform if training_transform is not None
        else dataset_transform
    )

    sampling_groups = config.get("sampling_groups", False)

    val_subsample = config.get('val_subsample', None)
    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)
    image_size = config.get('image_size', 224)
    num_workers = config.get('num_workers', 8)
    dataset_size = config.get('dataset_size', None)
    augment_data = config.get('augment_data', False)
    batch_size = config.get('batch_size', 32)

    n_concepts = len(CONCEPT_SEMANTICS)
    concept_group_map = {}

    if sampling_percent != 1:
        # Do the subsampling
        if sampling_groups:
            new_n_groups = int(np.ceil(len(concept_group_map) * sampling_percent))
            selected_groups_file = os.path.join(
                root_dir,
                f"selected_groups_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_groups_file):
                selected_groups = np.load(selected_groups_file)
            else:
                selected_groups = sorted(
                    np.random.permutation(len(concept_group_map))[:new_n_groups]
                )
                np.save(selected_groups_file, selected_groups)
            selected_concepts = []
            group_concepts = [x[1] for x in concept_group_map.items()]
            for group_idx in selected_groups:
                selected_concepts.extend(group_concepts[group_idx])
            selected_concepts = sorted(set(selected_concepts))
        else:
            new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
            selected_concepts_file = os.path.join(
                root_dir,
                f"selected_concepts_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_concepts_file):
                selected_concepts = np.load(selected_concepts_file)
            else:
                selected_concepts = sorted(
                    np.random.permutation(n_concepts)[:new_n_concepts]
                )
                np.save(selected_concepts_file, selected_concepts)
        # Then we also have to update the concept group map so that
        # selected concepts that were previously in the same concept
        # group are maintained in the same concept group
        new_concept_group = {}
        remap = dict((y, x) for (x, y) in enumerate(selected_concepts))
        selected_concepts_set = set(selected_concepts)
        for selected_concept in selected_concepts:
            for concept_group_name, group_concepts in concept_group_map.items():
                if selected_concept in group_concepts:
                    if concept_group_name in new_concept_group:
                        # Then we have already added this group
                        continue
                    # Then time to add this group!
                    new_concept_group[concept_group_name] = []
                    for other_concept in group_concepts:
                        if other_concept in selected_concepts_set:
                            # Add the remapped version of this concept
                            # into the concept group
                            new_concept_group[concept_group_name].append(
                                remap[other_concept]
                            )
        # And update the concept group map accordingly
        concept_group_map = new_concept_group
        print("\t\tSelected concepts:", selected_concepts)
        print(f"\t\tUpdated concept group map (with {len(concept_group_map)} groups):")
        for k, v in concept_group_map.items():
            print(f"\t\t\t{k} -> {v}")

        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]

        n_concepts = len(selected_concepts)
    else:
        concept_transform = None

    train_dl = load_data(
        split='train',
        batch_size=batch_size,
        image_size=image_size,
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=(
            training_transform if val_subsample is None else lambda x: x
        ),
        dataset_size=dataset_size,
        augment_data=augment_data,
        concept_transform=concept_transform,
    )

    if config.get('weight_loss', False):
        attribute_count = np.zeros((n_concepts,))
        samples_seen = 0
        for i, (_, y, c) in enumerate(train_dl):
            c = c.cpu().detach().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        imbalance = samples_seen / (attribute_count - 1 + 1e-8)
    else:
        imbalance = None

    if val_subsample is None:
        val_dl = load_data(
            split='val',
            batch_size=batch_size,
            image_size=image_size,
            root_dir=root_dir,
            num_workers=num_workers,
            dataset_transform=dataset_transform,
            dataset_size=dataset_size,
            augment_data=False,
            concept_transform=concept_transform,
        )
    else:
        ys = []
        xs = []
        attributes = []
        batch_size_opts = sorted(factors(len(train_dl.dataset)))
        batch_size = 1
        for opt in batch_size_opts:
            if opt < 256:
                batch_size = opt
            else:
                break
        fast_loader = torch.utils.data.DataLoader(
            train_dl.dataset,
            batch_size=batch_size,
            num_workers=2,
            drop_last=False,
            shuffle=False,
        )
        for x, y, attribute in fast_loader:
            y = y.detach().cpu().numpy()
            ys.append(y)
            x = x.detach().cpu().numpy()
            xs.append(x)
            if attribute is not None:
                attr = attribute.detach().cpu().numpy()
                attributes.append(attr)
        ys = np.concatenate(ys, axis=0)
        xs = np.concatenate(xs, axis=0)
        if attributes:
            attributes = np.concatenate(attributes, axis=0)
            x_train, x_val, y_train, y_val, c_train, c_val = \
                sklearn.model_selection.train_test_split(
                    xs,
                    ys,
                    attributes,
                    test_size=val_subsample,
                )
            train_dl = DataLoader(
                training_transform(torch.utils.data.TensorDataset(
                    torch.FloatTensor(x_train),
                    torch.LongTensor(y_train),
                    torch.FloatTensor(c_train),
                )),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
            )
            val_dl = DataLoader(
                dataset_transform(torch.utils.data.TensorDataset(
                    torch.FloatTensor(x_val),
                    torch.LongTensor(y_val),
                    torch.FloatTensor(c_val),
                )),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )
        else:
            x_train, x_val, y_train, y_val = \
                sklearn.model_selection.train_test_split(
                    xs,
                    ys,
                    test_size=val_subsample,
                )
            train_dl = DataLoader(
                training_transform(torch.utils.data.TensorDataset(
                    torch.FloatTensor(x_train),
                    torch.LongTensor(y_train),
                )),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
            )
            val_dl = DataLoader(
                dataset_transform(torch.utils.data.TensorDataset(
                    torch.FloatTensor(x_val),
                    torch.LongTensor(y_val),
                )),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )

    test_dl = load_data(
        split='test',
        batch_size=batch_size,
        image_size=image_size,
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=dataset_transform,
        dataset_size=dataset_size,
        augment_data=False,
        concept_transform=concept_transform,
    )

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (n_concepts, get_num_labels(**config), concept_group_map),
    )