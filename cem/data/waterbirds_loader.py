"""
Dataloader for the Waterbirds dataset by Sagawa and Koh et al. (ICLR 2020).

Adapted from https://github.com/kohpangwei/group_DRO/blob/master/data/cub_dataset.py
"""
import numpy as np
import os
import pandas as pd
import sklearn.model_selection
import torch
import torchvision.transforms as transforms

from functools import reduce
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, Subset, DataLoader

from cem.data.CUB200.cub_loader import CONCEPT_GROUP_MAP, SELECTED_CONCEPTS

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 2


# IMPORANT NOTE: THIS DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE BEING ABLE
#                TO RUN ANY CUB EXPERIMENTS!!
#                Instructions on how to download it can be found
#                in the original CBM paper's repository
#                found here: https://github.com/yewsiang/ConceptBottleneck
# CAN BE OVERWRITTEN WITH AN ENV VARIABLE DATASET_DIR
DATASET_DIR = os.environ.get("DATASET_DIR", 'data/waterbirds/')

########################################################
## Dataset Loader
########################################################

def attr_line_to_val(line):
    return int(line.split(" ")[2])

def get_sample_attributes(idx, lines):
    return np.array([
        attr_line_to_val(lines[x]) for x in range((idx - 1) * 312, idx * 312)
    ])


class WaterbirdsDataset(Dataset):
    """
    Waterbirds dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(
        self,
        root_dir,
        cub_root_dir=None,
        augment_data=False,
        split='train',
        image_size=224,
        class_dtype=float,
        use_attributes=True,
        concept_transform=None,
    ):
        self.root_dir = root_dir
        self.augment_data = augment_data
        self.split = split
        self.class_dtype = class_dtype
        self.cub_root_dir = cub_root_dir
        self.use_attributes = use_attributes
        self.concept_transform = concept_transform

        if not os.path.exists(self.root_dir):
            raise ValueError(
                f'{self.root_dir} does not exist yet. Please generate the '
                f'dataset first.'
            )

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.root_dir, 'metadata.csv')
        )

        # Get the y values
        self.y_array = self.metadata_df['y'].values.astype(np.float32)
        self.n_classes = N_CLASSES

        # We only support one confounder for CUB for now
        self.group_array = self.metadata_df['place'].values.astype(
            np.float32
        )

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        if self.cub_root_dir is not None:
            image_true_idx = self.metadata_df['img_id'].values
            with open(
                os.path.join(
                    self.cub_root_dir,
                    'CUB_200_2011/attributes/image_attribute_labels.txt',
                ),
                'r',
            ) as f:
                lines = [x.rstrip() for x in f]

            self.attributes = np.array([
                get_sample_attributes(idx, lines)[SELECTED_CONCEPTS] for idx in image_true_idx
            ])
        elif self.use_attributes:
            raise ValueError(
                'Unless cub_root_dir is provided, we cannot use image '
                'attributes for Waterbirds.'
            )

        self.split_array_map = []
        split_dict = {
            'train': 0,
            'val': 1,
            'test': 2,
        }
        for idx, split_val in enumerate(self.metadata_df['split'].values):
            if split_val == split_dict[self.split]:
                self.split_array_map.append(idx)

        if split == 'train':
            self.transform = get_transform_waterbirds(
                train=True,
                augment_data=augment_data,
                image_size=image_size,
            )
        else:
            self.transform = get_transform_waterbirds(
                train=False,
                augment_data=augment_data,
                image_size=image_size,
            )

    def __len__(self):
        return len(self.split_array_map)

    def __getitem__(self, idx):
        real_idx = self.split_array_map[idx]
        y = self.class_dtype(self.y_array[real_idx])
        if self.use_attributes:
            c = self.attributes[real_idx, :]
        else:
            c = np.array([self.group_array[real_idx]])
        if self.concept_transform is not None:
            c = self.concept_transform(c)

        img_filename = os.path.join(
            self.root_dir,
            self.filename_array[real_idx],
        )
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        c = torch.FloatTensor(c)
        return img, y, c


def get_transform_waterbirds(train, augment_data, image_size=224):
    """Helper function to get the appropiate transformation for the Waterbirds
    data loader.

    Args:
        train (bool): Whether or not this transform is for the training fold
            of the Waterbirds dataset or not.
        augment_data (bool): Whether or not we want to perform standard
            augmentations (crops and flips) used for the CUB dataset.
        image_size (int, optional): Size of the width and height of each
            of the generated images. Defaults to 224.

    Returns:
        torchvision.Transform: a valid torchvision transform to be applied to
            each image of the Waterbirds dataset being loaded.
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
    root_dir='data/waterbirds/',
    num_workers=1,
    dataset_transform=lambda x: x,
    dataset_size=None,
    augment_data=False,
    class_dtype=None,
    use_attributes=True,
    cub_root_dir=None,
    concept_transform=None,
):
    """Generates a Dataloader for the Waterbirds dataset.

    Args:
        split (str): Split of data to use for this loader. Must be one of
            'train', 'val', or 'test'.
        batch_size (int): Batch size used when loading data into the model.
        image_size (int, optional): Size of width and height of Waterbird
            images. Defaults to 299.
        root_dir (str, optional): Valid path to where the CUB/Waterbirds data
            is stored. Defaults to 'data/waterbirds/'.
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
            Waterbirds split.
    """
    dataset = WaterbirdsDataset(
        split=split,
        root_dir=root_dir,
        cub_root_dir=cub_root_dir,
        augment_data=augment_data,
        image_size=image_size,
        class_dtype=class_dtype,
        use_attributes=use_attributes,
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
        int: the number of class labels used in the waterbirds dataset.
    """
    return N_CLASSES

def get_num_attributes(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: the number of attributes in the waterbirds dataset.
    """
    # We return 2 as we have land backgrounds and water backgrounds.
    return 2

def generate_data(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
    dataset_transform=lambda x: x,
    val_subsample=None,
    class_dtype=None,
    training_transform=None,
):
    if root_dir is None:
        root_dir = DATASET_DIR
    output_classes = 2
    if not (class_dtype is None):
        class_dtype = getattr(np, class_dtype)
    else:
        class_dtype =  (int if output_classes > 1 else float)
    seed_everything(seed)

    training_transform = (
        training_transform if training_transform is not None
        else dataset_transform
    )

    sampling_groups = config.get("sampling_groups", False)

    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)
    image_size = config.get('image_size', 224)
    num_workers = config.get('num_workers', 8)
    dataset_size = config.get('dataset_size', None)
    augment_data = config.get('augment_data', False)
    use_attributes = config.get('use_attributes', True)
    batch_size = config.get('batch_size', 32)

    if use_attributes:
        n_concepts = len(SELECTED_CONCEPTS)
        concept_group_map = CONCEPT_GROUP_MAP.copy()
    else:
        # Then the background is used as an actual concept
        n_concepts = 2
        concept_group_map = None


    if sampling_percent != 1:
        # Do the subsampling
        if sampling_groups:
            new_n_groups = int(np.ceil(len(concept_group_map) * sampling_percent))
            selected_groups_file = os.path.join(
                DATASET_DIR,
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
                DATASET_DIR,
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
        class_dtype=class_dtype,
        cub_root_dir=config.get('cub_root_dir', None),
        use_attributes=use_attributes,
        concept_transform=concept_transform,
    )

    if config.get('weight_loss', False):
        attribute_count = np.zeros((n_concepts,))
        samples_seen = 0
        for i, (_, y, c) in enumerate(train_dl):
            c = c.cpu().detach().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        imbalance = samples_seen / attribute_count - 1
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
            class_dtype=class_dtype,
            cub_root_dir=config.get('cub_root_dir', None),
            use_attributes=use_attributes,
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
                    torch.FloatTensor(y_train),
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
                    torch.FloatTensor(y_val),
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
                    torch.FloatTensor(y_train),
                )),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
            )
            val_dl = DataLoader(
                dataset_transform(torch.utils.data.TensorDataset(
                    torch.FloatTensor(x_val),
                    torch.FloatTensor(y_val),
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
        class_dtype=class_dtype,
        cub_root_dir=config.get('cub_root_dir', None),
        use_attributes=use_attributes,
        concept_transform=concept_transform,
    )

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (n_concepts, get_num_labels(), concept_group_map),
    )