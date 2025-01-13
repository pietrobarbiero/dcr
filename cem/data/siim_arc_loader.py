"""
Dataloader the SIIM-ARC Dataset
"""
import numpy as np
import os
import torch

from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, Subset, DataLoader

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 1
N_CONCEPTS = 19
DATASET_DIR = os.environ.get("DATASET_DIR", 'data/siim_arc/')

########################################################
## Dataset Loader
########################################################

class SiimArcDataset(Dataset):
    """
    Siim-Arc dataset
    """

    def __init__(
        self,
        root_dir,
        split='train',
        concept_transform=None,
        additional_sample_transform=None,
    ):
        self.root_dir = root_dir
        self.split = split
        assert split in ['train', 'test', 'val']
        if concept_transform is None:
            concept_transform = lambda x: x
        self.concept_transform = concept_transform

        if not os.path.exists(self.root_dir):
            raise ValueError(
                f'{self.root_dir} does not exist yet. Please generate the '
                f'dataset first.'
            )

        self.transform = additional_sample_transform
        self._embs, self._concepts, self._labels = self._process_xray_concepts(
            split=split
        )

    def _process_xray_concepts(self, split='train'):
        x = torch.load(f'x_{split}.pt').float()
        c = torch.load(f'c_{split}.pt').float()
        embeddings = c.numpy()

        if not os.path.exists(f'c_{split}_bool.pt'):
            # Initialize an array to hold the cluster assignments
            cluster_assignments = np.zeros_like(embeddings)

            # Iterate over each dimension
            for i in range(embeddings.shape[1]):
                print('Clustering dimension ', i)
                # Reshape the dimension to be a 2D array (n, 1)
                dimension_data = embeddings[:, i].reshape(-1, 1)

                # Apply k-means clustering with 2 clusters
                kmeans = KMeans(n_clusters=2, random_state=0)
                kmeans.fit(dimension_data)

                # Get the cluster labels
                labels = kmeans.labels_

                # Calculate the mean of each cluster
                cluster_means = [dimension_data[labels == j].mean() for j in range(2)]

                # Determine which cluster has the lower mean and which has the higher mean
                if cluster_means[0] < cluster_means[1]:
                    cluster_assignments[:, i] = labels
                else:
                    cluster_assignments[:, i] = 1 - labels

            c = torch.tensor(cluster_assignments).float()
            torch.save(c, f'c_{split}_bool.pt')
        else:
            c = torch.load(f'c_{split}_bool.pt')
        y = torch.tensor(torch.load(f'y_true_{split}.pt'))
        y = torch.nn.functional.one_hot(y, num_classes=2).float()
        return x, c, y

    def __len__(self):
        return self._embs.shape[0]

    def __getitem__(self, idx):
        x = self._embs[idx]
        y = self._labels[idx]
        c = self._concepts[idx]
        if self.concept_transform is not None:
            x = self.concept_transform(c)
        if self.transform is not None:
            x = self.transform(x)
        return x, y, c

def load_data(
    split,
    batch_size,
    root_dir='data/siim_arc/',
    num_workers=1,
    dataset_transform=lambda x: x,
    dataset_size=None,
    concept_transform=None,
    additional_sample_transform=None,
):
    """
    TODO
    """
    dataset = SiimArcDataset(
        split=split,
        root_dir=root_dir,
        concept_transform=concept_transform,
        additional_sample_transform=additional_sample_transform,
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
    return N_CLASSES

def get_num_attributes(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: the number of attributes in the waterbirds dataset.
    """
    return N_CONCEPTS

def generate_data(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
    dataset_transform=lambda x: x,
    training_transform=None,

    train_sample_transform=None,
    test_sample_transform=None,
    val_sample_transform=None,
):
    if root_dir is None:
        root_dir = DATASET_DIR
    seed_everything(seed)

    training_transform = (
        training_transform if training_transform is not None
        else dataset_transform
    )


    sampling_percent = config.get("sampling_percent", 1)
    num_workers = config.get('num_workers', 8)
    dataset_size = config.get('dataset_size', None)
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
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=training_transform,
        dataset_size=dataset_size,
        concept_transform=concept_transform,
        additional_sample_transform=train_sample_transform,
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
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=dataset_transform,
        dataset_size=dataset_size,
        concept_transform=concept_transform,
        additional_sample_transform=val_sample_transform,
    )

    test_dl = load_data(
        split='test',
        batch_size=batch_size,
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=dataset_transform,
        dataset_size=dataset_size,
        concept_transform=concept_transform,
        additional_sample_transform=test_sample_transform,
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
