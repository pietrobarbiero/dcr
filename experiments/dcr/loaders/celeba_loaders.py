import argparse
import copy
import joblib
import numpy as np
import os
import torch
import torchvision
import logging
import io


from pathlib import Path
from pytorch_lightning import seed_everything
from torchvision import transforms

import cem.train.training as training
import cem.train.utils as utils
import cem.train.intervention_utils as intervention_utils
from dcr.models import CemEmbedder
import pytorch_lightning as pl

###############################################################################
## GLOBAL VARIABLES
###############################################################################

SELECTED_CONCEPTS = [
    2,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    39,
]

CONCEPT_SEMANTICS = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
]

# IMPORANT NOTE: THIS DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE BEING ABLE
#                TO RUN ANY CUB EXPERIMENTS!!
#                Instructions on how to download it can be found
#                in https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
CELEBA_ROOT = '.'


###############################################################################
## LOADERS
###############################################################################

def generate_celeba_cem_dataset(
    config,
    data_dir=CELEBA_ROOT,
    selected_concepts=SELECTED_CONCEPTS,
):
    use_binary_vector_class = config.get('use_binary_vector_class', False)
    if use_binary_vector_class:
        # Now reload by transform the labels accordingly
        width = config.get('label_binary_width', 5)
        def _binarize(concepts, selected, width):
            result = []
            binary_repr = []
            concepts = concepts[selected]
            for i in range(0, concepts.shape[-1], width):
                binary_repr.append(
                    str(int(np.sum(concepts[i : i + width]) > 0))
                )
            return int("".join(binary_repr), 2)

        celeba_train_data = torchvision.datasets.CelebA(
            root=data_dir,
            split='all',
            download=True,
            target_transform=lambda x: x[0].long() - 1,
            target_type=['attr'],
        )

        concept_freq = np.sum(
            celeba_train_data.attr.cpu().detach().numpy(),
            axis=0
        ) / celeba_train_data.attr.shape[0]
        print("Concept frequency is:", concept_freq)
        sorted_concepts = list(map(
            lambda x: x[0],
            sorted(enumerate(np.abs(concept_freq - 0.5)), key=lambda x: x[1]),
        ))
        num_concepts = config.get(
            'num_concepts',
            celeba_train_data.attr.shape[-1],
        )
        concept_idxs = sorted_concepts[:num_concepts]
        concept_idxs = sorted(concept_idxs)
        if config.get('num_hidden_concepts', 0):
            num_hidden = config.get('num_hidden_concepts', 0)
            hidden_concepts = sorted(
                sorted_concepts[
                    num_concepts:min(
                        (num_concepts + num_hidden),
                        len(sorted_concepts)
                    )
                ]
            )
        else:
            hidden_concepts = []
        print("Selecting concepts:", concept_idxs)
        print("\tAnd hidden concepts:", hidden_concepts)
        celeba_train_data = torchvision.datasets.CelebA(
            root=data_dir,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(config['image_size']),
                transforms.CenterCrop(config['image_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    _binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=['identity', 'attr'],
        )
        label_remap = {}
        vals, counts = np.unique(
            list(map(
                lambda x: _binarize(
                    x.cpu().detach().numpy(),
                    selected=(concept_idxs + hidden_concepts),
                    width=width,
                ),
                celeba_train_data.attr
            )),
            return_counts=True,
        )
        for i, label in enumerate(vals):
            label_remap[label] = i

        celeba_train_data = torchvision.datasets.CelebA(
            root=data_dir,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(config['image_size']),
                transforms.CenterCrop(config['image_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    label_remap[_binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    )],
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=['identity', 'attr'],
        )
        num_classes = len(label_remap)

        # And subsample to reduce its massive size
        factor = config.get('label_dataset_subsample', 1)
        if factor != 1:
            train_idxs = np.random.choice(
                np.arange(0, len(celeba_train_data)),
                replace=False,
                size=len(celeba_train_data)//factor,
            )
            print("Subsampling to", len(train_idxs), "elements.")
            celeba_train_data = torch.utils.data.Subset(
                celeba_train_data,
                train_idxs,
            )
    else:
        concept_selection = list(range(0, len(CONCEPT_SEMANTICS)))
        if config.get('selected_concepts', False):
            concept_selection = selected_concepts
        celeba_train_data = torchvision.datasets.CelebA(
            root=data_dir,
            split='all',
            download=True,
            target_transform=lambda x: x[0].long() - 1,
            target_type=['identity'],
        )
        vals, counts = np.unique(
            celeba_train_data.identity,
            return_counts=True,
        )
        sorted_labels = list(map(
            lambda x: x[0],
            sorted(zip(vals, counts), key=lambda x: -x[1])
        ))
        print(
            "Selecting",
            config['num_classes'],
            "out of",
            len(vals),
            "classes",
        )
        if result_dir:
            Path(result_dir).mkdir(parents=True, exist_ok=True)
            np.save(
                os.path.join(
                    result_dir,
                    f"selected_top_{config['num_classes']}_labels.npy",
                ),
                sorted_labels[:config['num_classes']],
            )
        label_remap = {}
        for i, label in enumerate(sorted_labels[:config['num_classes']]):
            label_remap[label] = i
        print("len(label_remap) =", len(label_remap))

        # Now reload by transform the labels accordingly
        celeba_train_data = torchvision.datasets.CelebA(
            root=data_dir,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(config['image_size']),
                transforms.CenterCrop(config['image_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    # If it is not in our map, then we make it be the token label
                    # config['num_classes'] which will be removed afterwards
                    label_remap.get(
                        x[0].cpu().detach().item() - 1,
                        config['num_classes']
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_selection].float(),
            ],
            target_type=['identity', 'attr'],
        )
        num_classes = config['num_classes']

        train_idxs = np.where(
            list(map(
                lambda x: x.cpu().detach().item() - 1 in label_remap,
                celeba_train_data.identity
            ))
        )[0]
        celeba_train_data = torch.utils.data.Subset(
            celeba_train_data,
            train_idxs,
        )
    total_samples = len(celeba_train_data)
    train_samples = int(0.7 * total_samples)
    test_samples = int(0.2 * total_samples)
    val_samples = total_samples - test_samples - train_samples
    print(
        f"Data split is: {total_samples} = {train_samples} (train) + "
        f"{test_samples} (test) + {val_samples} (validation)"
    )
    celeba_train_data, celeba_test_data, celeba_val_data = \
        torch.utils.data.random_split(
            celeba_train_data,
            [train_samples, test_samples, val_samples],
        )
    train_dl = torch.utils.data.DataLoader(
        celeba_train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )
    test_dl = torch.utils.data.DataLoader(
        celeba_test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    val_dl = torch.utils.data.DataLoader(
        celeba_val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    return train_dl, test_dl, val_dl