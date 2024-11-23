"""
Dataloader our synthetic traffic dataset
"""
import numpy as np
import os
import torch
import torchvision.transforms as transforms

from functools import reduce
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, Subset, DataLoader

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 1
DATASET_DIR = os.environ.get("DATASET_DIR", 'data/traffic/')

########################################################
## Dataset Loader
########################################################

class TrafficDataset(Dataset):
    """
    Synthetic traffic dataset
    """

    def __init__(
        self,
        root_dir,
        augment_data=False,
        split='train',
        image_size=256,
        concept_transform=None,
        class_dtype=float,
    ):
        self.root_dir = root_dir
        self.augment_data = augment_data
        self.split = split
        assert split in ['train', 'test', 'val']

        self.class_dtype = class_dtype
        if concept_transform is None:
            concept_transform = lambda x: x
        self.concept_transform = concept_transform

        if not os.path.exists(self.root_dir):
            raise ValueError(
                f'{self.root_dir} does not exist yet. Please generate the '
                f'dataset first.'
            )

        self.split_array_map = np.load(
            os.path.join(self.root_dir, f'{split}_indices.npy')
        )

        if split == 'train':
            self.transform = get_transform_traffic(
                train=True,
                augment_data=augment_data,
                image_size=image_size,
            )
        else:
            self.transform = get_transform_traffic(
                train=False,
                augment_data=augment_data,
                image_size=image_size,
            )

    def _from_meta_to_concepts(self, sample_meta):
        # Concepts will be:
        #  [0] Light color x-axis (0 if red, 1 if green)
        #  [1] Light color y-axis (0 if red, 1 if green)
        #  [2] Ambulance (1 if there is an ambulance in sight, 0 otherwise)
        #  [3] Car in intersection (1 if there is a car in the intersection,
        #      0 otherwise)
        #  [4] Other cars (1 if there are other cars visible anywhere, 0
        #      otherwise)
        #  [5] Selected car in north lane
        #  [6] Selected car in east lane
        #  [7] Selected car in south lane
        #  [8] Selected car in west lane
        #  [9] Green light on selected lane
        #  [10] Car perpendicular in intersection (1 if there is a car in the
        #      intersection in the direction perpendicular to this car, 0
        #      otherwise)
        #  [11] Ambulance Perpendicular (1 if the ambulance is in the
        #      direction perpendicular to the car, 0 otherwise)
        c = np.array([
            float(
                sample_meta['green'] and
                (sample_meta['selected_lane']['dir'] in ['east', 'west'])
            ), # [0]
            float(
                sample_meta['green'] and
                (sample_meta['selected_lane']['dir'] in ['south', 'north'])
            ), # [1]
            float(np.any(
                [x['ambulance'] for x in sample_meta['other_cars']]
            )), # [2]
            float(np.any([
                x['in_intersection']
                for x in sample_meta['other_cars']
            ])), # [3]
            float(len(sample_meta['other_cars']) > 0), # [4]
            float(sample_meta['selected_lane']['idx'] == 7), # [5]
            float(sample_meta['selected_lane']['idx'] == 1), # [6]
            float(sample_meta['selected_lane']['idx'] == 3), # [7]
            float(sample_meta['selected_lane']['idx'] == 5), # [8]
            float(sample_meta['green']), # [9]
            float(sample_meta['perp_intersection_occupied']), # [10]
            float(sample_meta['perp_incoming_ambulance']), # [11]

        ])
        return self.concept_transform(torch.FloatTensor(c))

    def _from_meta_to_label(self, sample_meta):
        y = self.class_dtype(sample_meta['action'] == 'continue')
        if self.class_dtype == float:
            y = torch.FloatTensor([y]).squeeze(-1)
        return y

    def sample_array(self, real_idx):
        sample_filename = os.path.join(
            self.root_dir,
            f'records/',
            f'sample_{real_idx}.npz'
        )
        loaded_data = np.load(sample_filename, allow_pickle=True)
        img = loaded_data['img']
        metadata = loaded_data['metadata'].item()
        img = torch.FloatTensor(
            # Transpose the image so that channels are first
            # Note: the image is already normalized so its values are within
            #       [0, 1]
            np.transpose(img, [2, 0, 1])
        )
        img = self.transform(img)
        return img, metadata

    def __len__(self):
        return len(self.split_array_map)

    def __getitem__(self, idx):
        real_idx = self.split_array_map[idx]
        img, sample_meta = self.sample_array(real_idx)
        y = self._from_meta_to_label(sample_meta)
        c = self._from_meta_to_concepts(sample_meta)
        return img, y, c


def get_transform_traffic(train, augment_data, image_size=256):
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
        transform = lambda x: x
        # transforms.Compose([
        #     transforms.Resize((
        #         int(image_size*scale),
        #         int(image_size*scale),
        #     )),
        #     # transforms.ToTensor(),
        # ])
    else:
        transform = lambda x: x
        # transform = transforms.Compose([
        #     transforms.Resize((
        #         int(image_size*scale),
        #         int(image_size*scale),
        #     )),
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.ToTensor(),
        # ])
    return transform


def load_data(
    split,
    batch_size,
    image_size=256,
    root_dir='data/traffic/',
    num_workers=1,
    dataset_transform=lambda x: x,
    dataset_size=None,
    augment_data=False,
    concept_transform=None,
):
    """
    TODO
    """
    dataset = TrafficDataset(
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

def get_num_labels(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: the number of class labels used in the waterbirds dataset.
    """
    return kwargs.get('n_classes', 1)

def get_num_attributes(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: the number of attributes in the waterbirds dataset.
    """
    return 10

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


    sampling_percent = config.get("sampling_percent", 1)
    image_size = config.get('image_size', 256)
    num_workers = config.get('num_workers', 8)
    dataset_size = config.get('dataset_size', None)
    augment_data = config.get('augment_data', False)
    batch_size = config.get('batch_size', 32)
    selected_concepts = config.get('selected_concepts', None)

    n_concepts = get_num_attributes(**config)
    concept_group_map = None

    if selected_concepts is not None:
        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]
        n_concepts = len(selected_concepts)
    elif sampling_percent != 1:
        # Do the subsampling
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
        dataset_transform=training_transform,
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
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None

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
