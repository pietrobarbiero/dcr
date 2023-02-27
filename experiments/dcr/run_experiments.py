import sys
sys.path.append(".")

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
import pytorch_lightning as pl


import cem.train.training as training
import cem.train.utils as utils
import cem.train.intervention_utils as intervention_utils
from dcr.models import CemEmbedder
import experiments.dcr.experiment_utils as experiment_utils
import experiments.dcr.loaders.celeba_loaders as celeba_loaders
import experiments.dcr.loaders.synth_loaders as synth_loaders
import experiments.dcr.loaders.cub_loaders as cub_loaders

###############################################################################
## PATHS FOR USED DATASETS
###############################################################################

# CHANGE ME IF YOU HAVE IT IN A DIFFERENT DIRECTORY OR PASS IT DIRECTLY
# AS AN ARGUMENT WHEN RUNNING THIS PROGRAM AS -p data_dir path/to/your/directory
CUB_DATA_DIR = '/homes/me466/UncertaintyIntervention/cem/data/CUB200/'

CELEBA_DATA_DIR = '.'

###############################################################################
## DEFAULT DATASETS CONFIGS
###############################################################################

def _default_c_extractor_arch(n_features, output_dim):
    return torch.nn.Sequential(*[
        torch.nn.Flatten(),
        torch.nn.Linear(n_features, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, output_dim),
    ])

DEFAULT_CONFIGS = {
    'celeba_cem': dict(
        cv=5,
        max_epochs=200,
        patience=15,
        batch_size=512,
        emb_size=16,
        extra_dims=0,
        concept_loss_weight=1,
        normalize_loss=False,
        learning_rate=0.005,
        weight_decay=4e-05,
        weight_loss=False,
        pretrain_model=True,
        c_extractor_arch="resnet34",
        optimizer="sgd",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        top_k_accuracy=[3, 5, 10],
        
        image_size=64,
        num_classes=1000,
        num_concepts=6,
        use_imbalance=True,
        use_binary_vector_class=True,
        label_binary_width=1,
        label_dataset_subsample=12,
        num_hidden_concepts=2,
        selected_concepts=False,
        
        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        embeding_activation=None,
        concat_prob=False,
        data_root=CELEBA_DATA_DIR,
        
        temperature=1,
    ),
    
    'cub': dict(
        cv=5,
        max_epochs=300,
        patience=15,
        batch_size=128,
        emb_size=16,
        extra_dims=0,
        concept_loss_weight=5,
        normalize_loss=False,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=True,
        pretrain_model=True,
        c_extractor_arch="resnet34",
        optimizer="sgd",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        sampling_percent=1,
        
        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        embeding_activation=None,
        concat_prob=False,
    ),
    
    'xor': dict(
        cv=5,
        dataset_size=3000,
        max_epochs=500,
        patience=15,
        batch_size=256,
        num_workers=8,
        emb_size=128,
        extra_dims=0,
        concept_loss_weight=1,
        normalize_loss=False,
        learning_rate=0.01,
        weight_decay=0,
        scheduler_step=20,
        weight_loss=False,
        optimizer="adam",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        masked=False,
        check_val_every_n_epoch=30,
        linear_c2y=True,
        embeding_activation="leakyrelu",
        
        
        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        concat_prob=False,
    ),
    
    'trig': dict(
        cv=5,
        dataset_size=3000,
        max_epochs=500,
        patience=15,
        batch_size=256,
        num_workers=8,
        emb_size=128,
        extra_dims=0,
        concept_loss_weight=1,
        normalize_loss=False,
        learning_rate=0.01,
        weight_decay=0,
        scheduler_step=20,
        weight_loss=False,
        optimizer="adam",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        masked=False,
        check_val_every_n_epoch=30,
        linear_c2y=True,
        embeding_activation="leakyrelu",
        
        
        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        concat_prob=False,
    ),
    
    'dot': dict(
        cv=5,
        dataset_size=3000,
        max_epochs=500,
        patience=15,
        batch_size=256,
        num_workers=8,
        emb_size=128,
        extra_dims=0,
        concept_loss_weight=1,
        normalize_loss=False,
        learning_rate=0.01,
        weight_decay=0,
        scheduler_step=20,
        weight_loss=False,
        optimizer="adam",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        masked=False,
        check_val_every_n_epoch=30,
        linear_c2y=True,
        embeding_activation="leakyrelu",
        
        
        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        concat_prob=False,
    ),
}

###############################################################################
## MAIN EXPERIMENT LOOP
###############################################################################

def main(
    result_dir,
    dataset_name,
    rerun=False,
    project_name='',
    num_workers=8,
    data_dir=None,
    global_params=None,
    dump_end_acts=False,
):
    # Set up seeds everywhere for the sake of reproducibility
    seed_everything(42)
    
    # Set up the config dictionary
    og_config = DEFAULT_CONFIGS.get(
        dataset_name,
        {},
    )
    og_config['num_workers'] = num_workers
    og_config['dataset_name'] = dataset_name
    if data_dir is not None:
        og_config['data_dir'] = data_dir
    utils.extend_with_global_params(og_config, global_params or [])
    
    # We will need groups and intervention frequency for when we compute interventions
    concept_group_map = None
    intervened_groups = None
    if dataset_name.lower() == "celeba_cem":
        data_dir = og_config.get('data_dir', CELEBA_DATA_DIR)
        data_loader = lambda config: celeba_loaders.generate_celeba_cem_dataset(
            config,
            data_dir=data_dir,
        )
    elif dataset_name.lower() == "cub":
        concept_group_map = cub_loaders.CUB_CONCEPT_GROUP_MAP
        intervened_groups = list(range(0, len(concept_group_map) + 1, 1))
        data_dir = og_config.get('data_dir', CUB_DATA_DIR)
        data_loader = lambda config: cub_loaders.generate_cub_dataset(
            config,
            data_dir=data_dir,
        )
    elif dataset_name.lower() == "xor":
        data_loader = lambda config: synth_loaders.generate_synth_dataset(config, 'xor')
    elif dataset_name.lower() == "trig":
        data_loader = lambda config: synth_loaders.generate_synth_dataset(config, 'trig')
    elif dataset_name.lower() == "dot":
        data_loader = lambda config: synth_loaders.generate_synth_dataset(config, 'dot')
        
    train_dl, test_dl, val_dl = data_loader(
        config=og_config,
    )
    
    # Then let's save the testing data for further analysis later on
    activations_dir = os.path.join(result_dir, "test_embedding_acts")
    Path(activations_dir).mkdir(parents=True, exist_ok=True)

    for (ds, name) in [
        (train_dl, "train"),
        (test_dl, "test"),
        (val_dl, "val"),
    ]:
        x_total = []
        y_total = []
        c_total = []
        for item in ds:
            if len(item) == 2:
                x, (y, c) = item
            else:
                x, y, c = item
            x_total.append(x.cpu().detach())
            y_total.append(y.cpu().detach())
            c_total.append(c.cpu().detach())
        x_inputs = np.concatenate(x_total, axis=0)
        print(f"x_{name}.shape =", x_inputs.shape)
        np.save(os.path.join(activations_dir, f"x_{name}.npy"), x_inputs)

        y_inputs = np.concatenate(y_total, axis=0)
        print(f"y_{name}.shape =", y_inputs.shape)
        np.save(os.path.join(activations_dir, f"y_{name}.npy"), y_inputs)

        c_inputs = np.concatenate(c_total, axis=0)
        print(f"c_{name}.shape =", c_inputs.shape)
        np.save(os.path.join(activations_dir, f"c_{name}.npy"), c_inputs)

    label_set = set()
    sample = next(iter(train_dl))
    real_sample = []
    for derp in sample:
        if isinstance(derp, list):
            real_sample += derp
        else:
            real_sample.append(derp)
    sample = real_sample
    print("Sample has", len(sample), "elements.")
    for i, derp in enumerate(sample):
        print("Element", i, "has shape", derp.shape, "and type", derp.dtype)

    print("Training sample shape is:", sample[0].shape)
    print("Training label shape is:", sample[1].shape)
    print("Training concept shape is:", sample[2].shape)

    n_concepts, n_tasks = sample[2].shape[-1], og_config.get(
        'num_classes',
        len(np.unique(
            np.load(os.path.join(activations_dir, f"y_train.npy"))
        )),
    )

    attribute_count = np.zeros((n_concepts,))
    samples_seen = 0
    for i, item in enumerate(train_dl):
        if len(item) == 2:
            _, (y, c) = item
        else:
            _, y, c = item
        print("\rIn batch", i, "we have seen", len(label_set), "classes")
        c = c.cpu().detach().numpy()
        attribute_count += np.sum(c, axis=0)
        samples_seen += c.shape[0]
        for l in y.reshape(-1).cpu().detach():
            label_set.add(l.item())

    print("Found a total of", len(label_set), "classes")
    if og_config.get("use_imbalance", False):
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None
    print("Imbalance:", imbalance)
    
    if og_config.get('c_extractor_arch', None) is None:
        og_config['c_extractor_arch'] = lambda output_dim: _default_c_extractor_arch(
            n_features=np.prod(sample[0].shape[1:]),
            output_dim=output_dim,
        )

    
    os.makedirs(result_dir, exist_ok=True)
    old_results = {}
    if os.path.exists(os.path.join(result_dir, f'results.joblib')):
        old_results = joblib.load(
            os.path.join(result_dir, f'results.joblib')
        )
    results = {}
    for split in range(og_config["cv"]):
        print(f'Experiment {split+1}/{og_config["cv"]}')
        results[f'{split}'] = {}
        
        # CEM baseline
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptEmbeddingModel"
        config["extra_name"] = f""
        config['training_intervention_prob'] = 0.25
        config['emb_size'] = config['emb_size']
        cem_model, cem_test_results = \
            training.train_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                split=split,
                result_dir=result_dir,
                rerun=rerun,
                project_name=project_name,
                seed=split,
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            cem_model,
            cem_test_results,
        )
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
        results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
            intervention_utils.intervene_in_cbm(
                gpu=1,
                config=config,
                test_dl=test_dl,
                train_dl=train_dl,
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                result_dir=result_dir,
                imbalance=imbalance,
                split=split,
                adversarial_intervention=False,
                rerun=rerun,
                concept_group_map=concept_group_map,
                intervened_groups=intervened_groups,
                old_results=old_results.get(str(split), {}).get(
                    f'test_acc_y_ints_{full_run_name}'
                ),
            )
        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
        # And dump learnt embeddings
        if dump_end_acts:
            print("Evaluating model in train, test, and validation datasets...")
            experiment_utils.dumb_pos_neg_cem_embs(
                config=config,
                result_dir=result_dir,
                imbalance=imbalance,
                x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
                x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                activations_dir=activations_dir,
                x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
                gpu=1,
                batch_size=og_config['batch_size'],
                split=split,
                model_name=full_run_name,
            )
            experiment_utils.dump_end_embeddings(
                x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
                x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
                n_concepts=n_concepts,
                model=cem_model,
                activations_dir=activations_dir,
                x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
                gpu=1,
                batch_size=og_config['batch_size'],
                split=split,
                model_name=full_run_name,
            )
        cem_config = copy.deepcopy(config)
        
        
#         # DCR Joint Pretrained!
#         cem_model = intervention_utils.load_trained_model(
#             config=cem_config,
#             n_tasks=n_tasks,
#             n_concepts=n_concepts,
#             result_dir=result_dir,
#             split=split,
#             imbalance=imbalance,
#         )
#         config = copy.deepcopy(og_config)
#         config["architecture"] = "DeepConceptReasoner"
#         config["extra_name"] = f"Joint_Pretrained_DCR_HardLabels"
#         config['training_intervention_prob'] = 0.5 #config.get('training_intervention_prob', 0.25)
#         config['temperature'] = config.get('temperature', 1)
#         config['emb_size'] = config['emb_size']
#         config['max_epochs'] = 30
#         config['concept_embedder'] = CemEmbedder(
#             cem=cem_model,
#             n_concepts=n_concepts,
#             freeze_model=False,
#             soft_concept_labels=False, # <---- WE WILL BE USING HARD LABELS FOR DCR's INPUT
#             test_with_soft_labels=True,
#             training_intervention_prob=config['training_intervention_prob'],
#         )
#         config['per_class_models'] = True
#         config['early_stopping_monitor'] = "val_y_accuracy"
#         config['early_stopping_mode'] = "max"
#         dcr_joint_pretrain_model, dcr_joint_pretrain_test_results = \
#             training.train_model(
#                 n_concepts=n_concepts,
#                 n_tasks=n_tasks,
#                 config=config,
#                 train_dl=train_dl,
#                 val_dl=val_dl,
#                 test_dl=test_dl,
#                 split=split,
#                 result_dir=result_dir,
#                 rerun=rerun,
#                 project_name=project_name,
#                 seed=split,
#                 imbalance=imbalance,
#             )
#         training.update_statistics(
#             results[f'{split}'],
#             config,
#             dcr_joint_pretrain_model,
#             dcr_joint_pretrain_test_results,
#         )
#         full_run_name = (
#             f"{config['architecture']}{config.get('extra_name', '')}"
#         )
#         results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
#             intervention_utils.intervene_in_cbm(
#                 gpu=1,
#                 config=config,
#                 test_dl=test_dl,
#                 train_dl=train_dl,
#                 n_tasks=n_tasks,
#                 n_concepts=n_concepts,
#                 result_dir=result_dir,
#                 imbalance=imbalance,
#                 split=split,
#                 adversarial_intervention=False,
#                 rerun=rerun,
#                 concept_group_map=concept_group_map,
#                 intervened_groups=intervened_groups,
#                 old_results=old_results.get(str(split), {}).get(
#                     f'test_acc_y_ints_{full_run_name}'
#                 ),
#             )
#         joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
#         # And dump learnt embeddings
#         if dump_end_acts:
#             print("Evaluating model in train, test, and validation datasets...")
#             experiment_utils.dump_end_embeddings(
#                 model=dcr_joint_pretrain_model,
#                 n_concepts=n_concepts,
#                 x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
#                 x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
#                 activations_dir=activations_dir,
#                 x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
#                 gpu=1,
#                 batch_size=og_config['batch_size'],
#                 split=split,
#                 model_name=full_run_name,
#             )

#         # DCR Joint Pretrained!
#         cem_model = intervention_utils.load_trained_model(
#             config=cem_config,
#             n_tasks=n_tasks,
#             n_concepts=n_concepts,
#             result_dir=result_dir,
#             split=split,
#             imbalance=imbalance,
#         )
#         config = copy.deepcopy(og_config)
#         config["architecture"] = "DeepConceptReasoner"
#         config["extra_name"] = f"Joint_Pretrained"
#         config['training_intervention_prob'] = 0.5 #config.get('training_intervention_prob', 0.25)
#         config['temperature'] = config.get('temperature', 1)
#         config['emb_size'] = config['emb_size']
#         config['max_epochs'] = 30
#         config['concept_embedder'] = CemEmbedder(
#             cem=cem_model,
#             n_concepts=n_concepts,
#             freeze_model=False,
#             soft_concept_labels=True,
#             test_with_soft_labels=True,
#             training_intervention_prob=config['training_intervention_prob'],
#         )
#         config['per_class_models'] = True
#         config['early_stopping_monitor'] = "val_y_accuracy"
#         config['early_stopping_mode'] = "max"
#         dcr_joint_pretrain_model, dcr_joint_pretrain_test_results = \
#             training.train_model(
#                 n_concepts=n_concepts,
#                 n_tasks=n_tasks,
#                 config=config,
#                 train_dl=train_dl,
#                 val_dl=val_dl,
#                 test_dl=test_dl,
#                 split=split,
#                 result_dir=result_dir,
#                 rerun=rerun,
#                 project_name=project_name,
#                 seed=split,
#                 imbalance=imbalance,
#             )
#         training.update_statistics(
#             results[f'{split}'],
#             config,
#             dcr_joint_pretrain_model,
#             dcr_joint_pretrain_test_results,
#         )
#         full_run_name = (
#             f"{config['architecture']}{config.get('extra_name', '')}"
#         )
#         results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
#             intervention_utils.intervene_in_cbm(
#                 gpu=1,
#                 config=config,
#                 test_dl=test_dl,
#                 train_dl=train_dl,
#                 n_tasks=n_tasks,
#                 n_concepts=n_concepts,
#                 result_dir=result_dir,
#                 imbalance=imbalance,
#                 split=split,
#                 adversarial_intervention=False,
#                 rerun=rerun,
#                 concept_group_map=concept_group_map,
#                 intervened_groups=intervened_groups,
#                 old_results=old_results.get(str(split), {}).get(
#                     f'test_acc_y_ints_{full_run_name}'
#                 ),
#             )
#         joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
#         # And dump learnt embeddings
#         if dump_end_acts:
#             print("Evaluating model in train, test, and validation datasets...")
#             experiment_utils.dump_end_embeddings(
#                 model=dcr_joint_pretrain_model,
#                 n_concepts=n_concepts,
#                 x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
#                 x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
#                 activations_dir=activations_dir,
#                 x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
#                 gpu=1,
#                 batch_size=og_config['batch_size'],
#                 split=split,
#                 model_name=full_run_name,
#             )

            
#         # DCR Sequential
#         cem_model = intervention_utils.load_trained_model(
#             config=cem_config,
#             n_tasks=n_tasks,
#             n_concepts=n_concepts,
#             result_dir=result_dir,
#             split=split,
#             imbalance=imbalance,
#         )
#         config = copy.deepcopy(og_config)
#         config["architecture"] = "DeepConceptReasoner"
#         config["extra_name"] = f"Sequential"
#         config['training_intervention_prob'] = 0.25
#         config['temperature'] = config.get('temperature', 1.5)
#         config['emb_size'] = config['emb_size']
#         config['max_epochs'] = 30
#         config['concept_embedder'] = CemEmbedder(cem=cem_model, n_concepts=n_concepts)
#         config['per_class_models'] = True
#         config['early_stopping_monitor'] = "val_y_accuracy"
#         config['early_stopping_mode'] = "max"

#         dcr_sequential_model, dcr_sequential_test_results = \
#             training.train_model(
#                 n_concepts=n_concepts,
#                 n_tasks=n_tasks,
#                 config=config,
#                 train_dl=train_dl,
#                 val_dl=val_dl,
#                 test_dl=test_dl,
#                 split=split,
#                 result_dir=result_dir,
#                 rerun=rerun,
#                 project_name=project_name,
#                 seed=split,
#                 imbalance=imbalance,
#             )
#         training.update_statistics(
#             results[f'{split}'],
#             config,
#             dcr_sequential_model,
#             dcr_sequential_test_results,
#         )
#         full_run_name = (
#             f"{config['architecture']}{config.get('extra_name', '')}"
#         )
#         results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
#             intervention_utils.intervene_in_cbm(
#                 gpu=1,
#                 config=config,
#                 test_dl=test_dl,
#                 train_dl=train_dl,
#                 n_tasks=n_tasks,
#                 n_concepts=n_concepts,
#                 result_dir=result_dir,
#                 imbalance=imbalance,
#                 split=split,
#                 adversarial_intervention=False,
#                 rerun=rerun,
#                 concept_group_map=concept_group_map,
#                 intervened_groups=intervened_groups,
#                 old_results=old_results.get(str(split), {}).get(
#                     f'test_acc_y_ints_{full_run_name}'
#                 ),
#             )
#         joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
#         # And dump learnt embeddings
#         if dump_end_acts:
#             print("Evaluating model in train, test, and validation datasets...")
#             experiment_utils.dump_end_embeddings(
#                 model=dcr_sequential_model,
#                 n_concepts=n_concepts,
#                 x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
#                 x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
#                 activations_dir=activations_dir,
#                 x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
#                 gpu=1,
#                 batch_size=og_config['batch_size'],
#                 split=split,
#                 model_name=full_run_name,
#             )
        
#         # DCR Sequential
#         cem_model = intervention_utils.load_trained_model(
#             config=cem_config,
#             n_tasks=n_tasks,
#             n_concepts=n_concepts,
#             result_dir=result_dir,
#             split=split,
#             imbalance=imbalance,
#         )
#         config = copy.deepcopy(og_config)
#         config["architecture"] = "DeepConceptReasoner"
#         config["extra_name"] = f"Sequential_HardConcepts_PosNegEmbs"
#         config['training_intervention_prob'] = 0.25
#         config['temperature'] = config.get('temperature', 1.5)
#         config['emb_size'] = config['emb_size']
#         config['max_epochs'] = 30
#         config['concept_embedder'] = CemEmbedder(
#             cem=cem_model,
#             n_concepts=n_concepts,
#             freeze_model=True,
#             soft_concept_labels=False,
#             pass_pos_neg_embs=True,
#             test_with_soft_labels=False,  # <--- even at testing we assume we know ground truths
#         )
#         config['per_class_models'] = True
#         config['early_stopping_monitor'] = "val_y_accuracy"
#         config['early_stopping_mode'] = "max"

#         dcr_sequential_model, dcr_sequential_test_results = \
#             training.train_model(
#                 n_concepts=n_concepts,
#                 n_tasks=n_tasks,
#                 config=config,
#                 train_dl=train_dl,
#                 val_dl=val_dl,
#                 test_dl=test_dl,
#                 split=split,
#                 result_dir=result_dir,
#                 rerun=rerun,
#                 project_name=project_name,
#                 seed=split,
#                 imbalance=imbalance,
#             )
#         training.update_statistics(
#             results[f'{split}'],
#             config,
#             dcr_sequential_model,
#             dcr_sequential_test_results,
#         )
#         full_run_name = (
#             f"{config['architecture']}{config.get('extra_name', '')}"
#         )
#         results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
#             intervention_utils.intervene_in_cbm(
#                 gpu=1,
#                 config=config,
#                 test_dl=test_dl,
#                 train_dl=train_dl,
#                 n_tasks=n_tasks,
#                 n_concepts=n_concepts,
#                 result_dir=result_dir,
#                 imbalance=imbalance,
#                 split=split,
#                 adversarial_intervention=False,
#                 rerun=rerun,
#                 concept_group_map=concept_group_map,
#                 intervened_groups=intervened_groups,
#                 old_results=old_results.get(str(split), {}).get(
#                     f'test_acc_y_ints_{full_run_name}'
#                 ),
#             )
#         joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
#         # And dump learnt embeddings
#         if dump_end_acts:
#             print("Evaluating model in train, test, and validation datasets...")
#             experiment_utils.dump_end_embeddings(
#                 model=dcr_sequential_model,
#                 n_concepts=n_concepts,
#                 x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
#                 x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
#                 activations_dir=activations_dir,
#                 x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
#                 gpu=1,
#                 batch_size=og_config['batch_size'],
#                 split=split,
#                 model_name=full_run_name,
#             )
        
#         # DCR Sequential
#         cem_model = intervention_utils.load_trained_model(
#             config=cem_config,
#             n_tasks=n_tasks,
#             n_concepts=n_concepts,
#             result_dir=result_dir,
#             split=split,
#             imbalance=imbalance,
#         )
#         config = copy.deepcopy(og_config)
#         config["architecture"] = "DeepConceptReasoner"
#         config["extra_name"] = f"Sequential_SoftConcepts_PosNegEmbs"
#         config['training_intervention_prob'] = 0.25
#         config['temperature'] = config.get('temperature', 1.5)
#         config['emb_size'] = config['emb_size']
#         config['max_epochs'] = 30
#         config['concept_embedder'] = CemEmbedder(
#             cem=cem_model,
#             n_concepts=n_concepts,
#             freeze_model=True,
#             soft_concept_labels=True,
#             pass_pos_neg_embs=True,
#             test_with_soft_labels=True,
#         )
#         config['per_class_models'] = True
#         config['early_stopping_monitor'] = "val_y_accuracy"
#         config['early_stopping_mode'] = "max"

#         dcr_sequential_model, dcr_sequential_test_results = \
#             training.train_model(
#                 n_concepts=n_concepts,
#                 n_tasks=n_tasks,
#                 config=config,
#                 train_dl=train_dl,
#                 val_dl=val_dl,
#                 test_dl=test_dl,
#                 split=split,
#                 result_dir=result_dir,
#                 rerun=rerun,
#                 project_name=project_name,
#                 seed=split,
#                 imbalance=imbalance,
#             )
#         training.update_statistics(
#             results[f'{split}'],
#             config,
#             dcr_sequential_model,
#             dcr_sequential_test_results,
#         )
#         full_run_name = (
#             f"{config['architecture']}{config.get('extra_name', '')}"
#         )
#         results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
#             intervention_utils.intervene_in_cbm(
#                 gpu=1,
#                 config=config,
#                 test_dl=test_dl,
#                 train_dl=train_dl,
#                 n_tasks=n_tasks,
#                 n_concepts=n_concepts,
#                 result_dir=result_dir,
#                 imbalance=imbalance,
#                 split=split,
#                 adversarial_intervention=False,
#                 rerun=rerun,
#                 concept_group_map=concept_group_map,
#                 intervened_groups=intervened_groups,
#                 old_results=old_results.get(str(split), {}).get(
#                     f'test_acc_y_ints_{full_run_name}'
#                 ),
#             )
#         joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
#         # And dump learnt embeddings
#         if dump_end_acts:
#             print("Evaluating model in train, test, and validation datasets...")
#             experiment_utils.dump_end_embeddings(
#                 model=dcr_sequential_model,
#                 n_concepts=n_concepts,
#                 x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
#                 x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
#                 activations_dir=activations_dir,
#                 x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
#                 gpu=1,
#                 batch_size=og_config['batch_size'],
#                 split=split,
#                 model_name=full_run_name,
#             )
            
# #         # DCR Joint
# #         config = copy.deepcopy(og_config)
# #         config["architecture"] = "DeepConceptReasoner"
# #         config["extra_name"] = f"Joint"
# #         config['training_intervention_prob'] = 0.25
# #         config['temperature'] = config.get('temperature', 1)
# #         config['emb_size'] = config['emb_size']
# #         config['early_stopping_monitor'] = "val_y_accuracy"
# #         config['early_stopping_mode'] = "max"
# #         dcr_joint_model, dcr_joint_test_results = \
# #             training.train_model(
# #                 n_concepts=n_concepts,
# #                 n_tasks=n_tasks,
# #                 config=config,
# #                 train_dl=train_dl,
# #                 val_dl=val_dl,
# #                 test_dl=test_dl,
# #                 split=split,
# #                 result_dir=result_dir,
# #                 rerun=rerun,
# #                 project_name=project_name,
# #                 seed=split,
# #                 imbalance=imbalance,
# #             )
# #         training.update_statistics(
# #             results[f'{split}'],
# #             config,
# #             dcr_joint_model,
# #             dcr_joint_test_results,
# #         )
# #         full_run_name = (
# #             f"{config['architecture']}{config.get('extra_name', '')}"
# #         )
# #         results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
# #             intervention_utils.intervene_in_cbm(
# #                 gpu=1,
# #                 config=config,
# #                 test_dl=test_dl,
# #                 train_dl=train_dl,
# #                 n_tasks=n_tasks,
# #                 n_concepts=n_concepts,
# #                 result_dir=result_dir,
# #                 imbalance=imbalance,
# #                 split=split,
# #                 adversarial_intervention=False,
# #                 rerun=rerun,
# #                 concept_group_map=concept_group_map,
# #                 intervened_groups=intervened_groups,
# #                 old_results=old_results.get(str(split), {}).get(
# #                     f'test_acc_y_ints_{full_run_name}'
# #                 ),
# #             )
# #         joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
# #         # And dump learnt embeddings
# #         if dump_end_acts:
# #             print("Evaluating model in train, test, and validation datasets...")
# #             experiment_utils.dump_end_embeddings(
# #                 model=dcr_joint_model,
# #                 n_concepts=n_concepts,
# #                 x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
# #                 x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
# #                 activations_dir=activations_dir,
# #                 x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
# #                 gpu=1,
# #                 batch_size=og_config['batch_size'],
# #                 split=split,
# #                 model_name=full_run_name,
# #             )
        

#         # CBM Logit baseline
#         config = copy.deepcopy(og_config)
#         config["architecture"] = "ConceptBottleneckModel"
#         config["bool"] = False
#         config["extra_name"] = f"Logit"
#         config["bottleneck_nonlinear"] = "leakyrelu"
#         config["sigmoidal_prob"] = False
#         cbm_logit_model, cbm_logit_test_results = \
#             training.train_model(
#                 n_concepts=n_concepts,
#                 n_tasks=n_tasks,
#                 config=config,
#                 train_dl=train_dl,
#                 val_dl=val_dl,
#                 test_dl=test_dl,
#                 split=split,
#                 result_dir=result_dir,
#                 rerun=rerun,
#                 project_name=project_name,
#                 seed=split,
#                 imbalance=imbalance,
#             )
#         training.update_statistics(
#             results[f'{split}'],
#             config,
#             cbm_logit_model,
#             cbm_logit_test_results,
#         )
#         full_run_name = (
#             f"{config['architecture']}{config.get('extra_name', '')}"
#         )
#         results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
#             intervention_utils.intervene_in_cbm(
#                 gpu=1,
#                 config=config,
#                 test_dl=test_dl,
#                 train_dl=train_dl,
#                 n_tasks=n_tasks,
#                 n_concepts=n_concepts,
#                 result_dir=result_dir,
#                 imbalance=imbalance,
#                 split=split,
#                 adversarial_intervention=False,
#                 rerun=rerun,
#                 concept_group_map=concept_group_map,
#                 intervened_groups=intervened_groups,
#                 old_results=old_results.get(str(split), {}).get(
#                     f'test_acc_y_ints_{full_run_name}'
#                 ),
#             )
#         joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
#         # And dump learnt embeddings
#         if dump_end_acts:
#             print("Evaluating model in train, test, and validation datasets...")
#             experiment_utils.dump_end_embeddings(
#                 model=cbm_logit_model,
#                 n_concepts=n_concepts,
#                 x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
#                 x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
#                 activations_dir=activations_dir,
#                 x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
#                 gpu=1,
#                 batch_size=og_config['batch_size'],
#                 split=split,
#                 model_name=full_run_name,
#             )
        
        # CBM Sigmoidal baseline
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["bool"] = False
        config["extra_name"] = f"Sigmoidal"
        config["bottleneck_nonlinear"] = "leakyrelu"
        config["sigmoidal_prob"] = True
        cbm_sigmoidal_model, cbm_sigmoidal_test_results = \
            training.train_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                split=split,
                result_dir=result_dir,
                rerun=rerun,
                project_name=project_name,
                seed=split,
                imbalance=imbalance,
            )
        training.update_statistics(
            results[f'{split}'],
            config,
            cbm_sigmoidal_model,
            cbm_sigmoidal_test_results,
        )
        
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
        results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
            intervention_utils.intervene_in_cbm(
                gpu=1,
                config=config,
                test_dl=test_dl,
                train_dl=train_dl,
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                result_dir=result_dir,
                imbalance=imbalance,
                split=split,
                adversarial_intervention=False,
                rerun=rerun,
                concept_group_map=concept_group_map,
                intervened_groups=intervened_groups,
                old_results=old_results.get(str(split), {}).get(
                    f'test_acc_y_ints_{full_run_name}'
                ),
            )
        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))
        # And dump learnt embeddings
        if dump_end_acts:
            print("Evaluating model in train, test, and validation datasets...")
            experiment_utils.dump_end_embeddings(
                model=cbm_sigmoidal_model,
                n_concepts=n_concepts,
                x_train=np.load(os.path.join(activations_dir, f"x_train.npy")),
                x_test=np.load(os.path.join(activations_dir, f"x_test.npy")),
                activations_dir=activations_dir,
                x_val=np.load(os.path.join(activations_dir, f"x_val.npy")),
                gpu=1,
                batch_size=og_config['batch_size'],
                split=split,
                model_name=full_run_name,
            )

    return results


###############################################################################
## ENTRY POINT
###############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Runs our benchmarking of DCR vs different baselines.'
        ),
    )
    parser.add_argument(
        'dataset',
        default='',
        help=(
            "Name of the dataset to be used for evaluation."
        ),
        choices=['xor', 'trig', 'dot', 'cub', 'celeba_cem'],
        metavar="dataset_name",

    )
    parser.add_argument(
        '--project_name',
        default='',
        help=(
            "Project name used for Weights & Biases monitoring. If not "
            "provided, then we will assume we do not run a w&b project."
        ),
        metavar="name",

    )

    parser.add_argument(
        '--output_dir',
        '-o',
        default=None,
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use ./results/<dataset_name>/."
        ),
        metavar="path",

    )
    parser.add_argument(
        '--rerun',
        '-r',
        default=False,
        action="store_true",
        help=(
            "If set, then we will force a rerun of the entire experiment even if "
            "valid results are found in the provided output directory. Note that "
            "this may overwrite and previous results, so use with care."
        ),

    )
    parser.add_argument(
        '--num_workers',
        default=8,
        help=(
            'number of workers used for data feeders. Do not use more workers '
            'than cores in the machine.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        '--data_dir',
        default=None,
        help=(
            'directory containing the dataset of interest. If not given, '
            'then we will default to the one defined in this file for the '
            'dataset.'
        ),
        metavar='path',
        type=str,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )
    parser.add_argument(
        "-a",
        "--dump_end_acts",
        action="store_true",
        default=False,
        help=(
            "if given, it will dump the learnt embeddings of each method and fold into "
            "the output directory."
        ),
    )
    parser.add_argument(
        '-p',
        '--param',
        action='append',
        nargs=2,
        metavar=('param_name=value'),
        help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
        ),
        default=[],
    )
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    if args.output_dir is None:
        args.output_dir = f'results/{args.dataset}'
    main(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        rerun=args.rerun,
        result_dir=args.output_dir,
        project_name=args.project_name,
        num_workers=args.num_workers,
        global_params=args.param,
        dump_end_acts=args.dump_end_acts,
    )
