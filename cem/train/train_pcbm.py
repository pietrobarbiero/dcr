
import copy
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from sklearn.svm import SVC
from tqdm import tqdm

import cem.models.standard as standard_models
import cem.train.evaluate as evaluate
import cem.train.utils as utils
import cem.utils.data as data_utils

from cem.models.base_wrappers import WrapperModule, EvalOnlyWrapperModule
from cem.models.construction import construct_model
from cem.train.training import _make_callbacks, _check_interruption, _restore_checkpoint

################################################################################
## Concept Embedding Learning
################################################################################


def get_cavs(train_embbedings, train_concept_labels, C):
    """
    Adapted from https://github.com/mertyg/post-hoc-cbm/blob/main/concepts/concept_utils.py
    """
    svm = SVC(C=C, kernel="linear")
    svm.fit(train_embbedings, train_concept_labels)
    return (svm.coef_, svm.intercept_)


# Keys:
# - pretrained_architecture (a config with the architecture of the blackbox model)
# - blackbox_path (path of weights for black-box model)
# - blackbox_early_stop_config (early stopping configuration for the black box model)
# - embedding_layer_name (name of the layer we will use in the black box model to extract the concepts)
# - svd_penalty (SVD l2 penalty)


################################################################################
## Training Pipeline
################################################################################


def train_pcbm(
    input_shape,
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl,
    run_name,
    result_dir=None,
    test_dl=None,
    split=None,
    imbalance=None,
    task_class_weights=None,
    rerun=False,
    logger=False,
    project_name='',
    seed=None,
    save_model=True,
    activation_freq=0,
    single_frequency_epochs=0,
    gradient_clip_val=0,
    old_results=None,
    enable_checkpointing=False,
    accelerator="auto",
    devices="auto",
    cav_train_dl=None,
):
    enable_checkpointing = (
        True if config.get('early_stopping_best_model', False)
        else enable_checkpointing
    )
    assert activation_freq == 0, (
        'PCBM training currently does not support activation dumping during '
        'training.'
    )
    if seed is not None:
        seed_everything(seed)

    if split is not None:
        full_run_name = (
            f"{run_name}_fold_{split + 1}"
        )
    else:
        full_run_name = (
            f"{run_name}"
        )
    print(f"[Training {run_name} (trial {split + 1})]")
    print("config:")
    for key, val in config.items():
        print(f"\t{key} -> {val}")

    # First create the original pretrained model
    model_saved_path = os.path.join(
        result_dir,
        f'{full_run_name}.pt'
    )

    if (project_name) and result_dir and (
        not os.path.exists(model_saved_path)
    ):
        # Lazy import to avoid importing unless necessary
        import wandb
        enter_obj = wandb.init(
            project=project_name,
            name=full_run_name,
            config=config,
            reinit=True
        )
        used_logger = WandbLogger(
            name=full_run_name,
            project=project_name,
            save_dir=os.path.join(result_dir, "logs"),
        ) if rerun or (not os.path.exists(model_saved_path)) else False
    else:
        enter_obj = utils.EmptyEnter()
        used_logger = logger or False
    trainer_args = dict(
        accelerator=accelerator,
        devices=devices,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
        enable_checkpointing=enable_checkpointing,
        gradient_clip_val=gradient_clip_val,
        # Only use the wandb logger when it is a fresh run
        logger=used_logger,
    )

    with enter_obj as run:
        num_epochs = []
        times = []

        #####################
        ## Step 1: load or train the blackbox model
        #####################

        print("\tConstructing black-box model")
        blackbox_model_config = config["blackbox_model_config"]
        bbox, _ = standard_models.construct_standard_model(
            architecture=blackbox_model_config['name'],
            input_shape=input_shape,
            n_labels=n_tasks,
            seed=seed,
            **blackbox_model_config,
        )
        if 'optimizer_config' in blackbox_model_config:
            bbox_optimizer_config = blackbox_model_config['optimizer_config']
        else:
            bbox_optimizer_config = dict(
                optimizer=config['optimizer'],
                learning_rate=config.get('learning_rate', 1e-3),
                weight_decay=config.get('weight_decay', 0),
                momentum=config.get('momentum', 0.9),
                lr_scheduler=dict(
                    type='reduce_on_plateau',

                )
            )
        if 'bbox_loss_config' in blackbox_model_config:
            bbox_loss_config = blackbox_model_config['bbox_loss_config']
        else:
            bbox_loss_config = dict(
                name="binary_cross_entropy" if n_tasks <= 2 else "cross_entropy"
            )
        bbox = WrapperModule(
            model=bbox,
            optimizer_config=bbox_optimizer_config,
            loss_config=bbox_loss_config,
            output_activation=None,
            metrics=["accuracy" if n_tasks > 2 else "auc"],
            task_class_weights=task_class_weights,
            n_labels=n_tasks,

        )
        print(
            "\t\t[Number of parameters in black-box model",
            sum(p.numel() for p in bbox.parameters() if p.requires_grad),
            "]"
        )
        print(
            "\t\t[Number of non-trainable parameters in black-box model",
            sum(p.numel() for p in bbox.parameters() if not p.requires_grad),
            "]",
        )
        bbox_path = config.get("blackbox_path")
        if bbox_path:
            bbox_path = os.path.join(result_dir, bbox_path)
            if bbox_path[-3:] != ".pt":
                bbox_path +=  ".pt"
        if (not rerun) and bbox_path and os.path.exists(bbox_path):
            print(f"\t\tLoading blackbox model's weights from {bbox_path}")
            bbox.load_state_dict(
                torch.load(bbox_path),
                strict=False,
            )
        else:
            bbox_epochs = config.get('blackbox_training_epochs', 0)
            if bbox_epochs != 0:
                start_time = time.time()
                print(
                    f"\t\tTraining blackbox model for {bbox_epochs} epochs"
                )
                bbox_callbacks, bbox_ckpt_call = _make_callbacks(
                    config.get('blackbox_early_stop_config', config),
                    result_dir,
                    ("BlackBoxModel_" + full_run_name),
                )
                bbox_trainer = pl.Trainer(
                    max_epochs=bbox_epochs,
                    callbacks=bbox_callbacks,
                    **trainer_args,
                )
                bbox_trainer.fit(bbox, train_dl, val_dl)
                _check_interruption(bbox_trainer)
                _restore_checkpoint(
                    model=bbox,
                    max_epochs=bbox_epochs,
                    ckpt_call=bbox_ckpt_call,
                    trainer=bbox_trainer,
                )
                num_epochs.append(bbox_trainer.current_epoch)
                times.append(time.time() - start_time)
                print(
                    f"\t\tDone with black box model training "
                    f"after {bbox_trainer.current_epoch} epochs ({times[-1]} sec)!"
                )
                if bbox_path:
                    print(f"\t\tSaving blackbox model's weights from {bbox_path}")
                    torch.save(
                        bbox.state_dict(),
                        bbox_path,
                    )
            else:
                num_epochs.append(0)
                times.append(0)


        #####################
        ## Step 2: extract embedding generator from black box model
        #####################
        print(f"\tConstructing embedding generator from black box model")
        _, latent_name = standard_models.get_out_layer_name_from_config(
            blackbox_model_config["name"],
            add_linear_layers=blackbox_model_config.get('add_linear_layers'),
        )
        embedding_generator = standard_models.create_feature_extractor(
            bbox.wrapped_model,
            return_nodes={latent_name: latent_name},
        )
        embedding_generator = EvalOnlyWrapperModule(
            model=embedding_generator,
            n_labels=n_tasks,
            metrics=[],
            output_activation=None,
            logits_name=latent_name,
        )


        #####################
        ## Step 3: learn CAVs from the black box's embeddings
        #####################

        print(f"\tLearning concept vectors from black box model")
        cav_path = os.path.join(
            result_dir,
            f'{full_run_name}_CAVs.npy'
        )
        cav_path = config.get('cav_path', cav_path)
        intercept_path = os.path.join(
            result_dir,
            f'{full_run_name}_Intercepts.npy'
        )
        intercept_path = config.get('intercept_path', intercept_path)
        if (not rerun) and os.path.exists(intercept_path) and (
            os.path.exists(cav_path)
        ):
            # Then let's simply load them!
            print(f"\t\tFound CAVs cached, so we will load them")

            concept_vectors = np.load(cav_path)
            cav_intercepts = np.load(intercept_path)
        else:
            start_time = time.time()
            # Time to extract them
            # First generate the training set for concept extraction
            print(f"\t\tLoading CAV training set...")
            cav_train_dl = train_dl if cav_train_dl is None else cav_train_dl
            x_train, _, c_train = data_utils.daloader_to_memory(
                cav_train_dl
            )

            # Then produce the embeddings for the entire training set
            prediction_trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                logger=False,
            )
            print(f"\t\tGenerating embeddings...")
            train_batch_embs = prediction_trainer.predict(
                embedding_generator,
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.from_numpy(x_train),
                    ),
                    batch_size=1,
                    num_workers=config.get('num_workers', 5),
                ),
            )
            train_embs = torch.concat(train_batch_embs, dim=0)
            print(f"\t\tConstructing CAVs...")
            concept_vectors, cav_intercepts = [], []
            for i in tqdm(range(n_concepts)):
                cav, intercept = get_cavs(
                    train_embbedings=train_embs,
                    train_concept_labels=c_train[:, i],
                    C=config.get('svd_penalty', 1),
                )
                concept_vectors.append(cav)
                cav_intercepts.append(intercept)
            concept_vectors = np.concatenate(concept_vectors, axis=0)
            cav_intercepts = np.concatenate(cav_intercepts, axis=0)
            if save_model:
                np.save(cav_path, concept_vectors)
                np.save(intercept_path, cav_intercepts)
            num_epochs.append(0)
            times.append(time.time() - start_time)
            print(f"\t\tDone learning CAVs after {times[-1]} seconds!")


        # CONSTRUCT THE PCBM!
        print(f"\tConstructing PCBM")

        # Initialize the actual final PCBM model by setting the CAVs and
        # backbone model extractor accordingly
        pcbm_config = copy.deepcopy(config)
        pcbm_config['concept_vectors'] = torch.FloatTensor(concept_vectors)
        pcbm_config['concept_vector_intercepts'] = torch.FloatTensor(cav_intercepts)
        pcbm_config['pretrained_model'] = embedding_generator.wrapped_model
        pcbm_config['emb_size'] = pcbm_config['concept_vectors'].shape[-1]
        torch.save(
            bbox.state_dict(),
                os.path.join(
                result_dir,
                f'{full_run_name}_Emb_Extractor.pt'
            ),
        )
        c2y_model = None
        if 'c2y_model' in pcbm_config:
            c2y_model_config = config["c2y_model_config"]
            c2y_model, _ = standard_models.construct_standard_model(
                architecture=c2y_model_config['name'],
                input_shape=(n_concepts,),
                n_labels=n_tasks,
                seed=seed,
                **c2y_model_config,
            )
        pcbm_config['c2y_model'] = c2y_model
        pcbm = construct_model(
            n_concepts,
            n_tasks,
            pcbm_config,
            imbalance=None,  # Not relevant cause there is no concept supervision
            task_class_weights=task_class_weights,
        )
        print("\tSetting up PCBM")
        print(
            "\t[Number of parameters in PCBM",
            sum(p.numel() for p in pcbm.parameters() if p.requires_grad),
            "]"
        )
        print(
            "\t[Number of non-trainable parameters in PCBM",
            sum(p.numel() for p in pcbm.parameters() if not p.requires_grad),
            "]",
        )

        if (not rerun) and os.path.exists(model_saved_path):
            print("\tFound cached model... loading it")
            pcbm.load_state_dict(torch.load(model_saved_path))
            if os.path.exists(
                model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [training_time, num_epochs] = np.load(
                    model_saved_path.replace(".pt", "_training_times.npy")
                )
            else:
                training_time, num_epochs  = 0, 0
        else:
            #####################
            ## Step 4: Train PCBM with CAVs
            #####################
            if pcbm.residual:
                pcbm.residual_use = False

            pcbm_epochs = config.get('max_epochs')
            print(f"\tTraining PCBM from scratch for {pcbm_epochs} epochs")
            start_time = time.time()

            pcbm_callbacks, pcbm_ckpt_call = _make_callbacks(
                config.get('pcbm_early_stop_config', config),
                result_dir,
                full_run_name,
            )
            pcbm_trainer = pl.Trainer(
                max_epochs=pcbm_epochs,
                callbacks=pcbm_callbacks,
                **trainer_args,
            )
            pcbm_trainer.fit(pcbm, train_dl, val_dl)
            _check_interruption(pcbm_trainer)
            _restore_checkpoint(
                model=pcbm,
                max_epochs=pcbm_epochs,
                ckpt_call=pcbm_ckpt_call,
                trainer=pcbm_trainer,
            )
            num_epochs.append(pcbm_trainer.current_epoch)
            times.append(time.time() - start_time)
            print(
                f"\t\tDone with PCBM training "
                f"after {pcbm_trainer.current_epoch} epochs ({times[-1]} sec)!"
            )


            #####################
            ## Step 5: Train Residuals
            #####################
            if pcbm.residual:
                print(f"\tTraining Residual Model")
                residual_epochs = config.get('max_residual_epochs', config['max_epochs'])

                # Set up the model accordingly so that it starts using the residual
                # output
                pcbm.residual_use = True
                pcbm.freeze_non_residual_components()
                print(
                    f"\t\tTraining PCBM's residual model from scratch "
                    f"for {residual_epochs} epochs"
                )
                start_time = time.time()

                residual_callbacks, residual_ckpt_call = _make_callbacks(
                    config.get('residual_early_stop_config', config),
                    result_dir,
                    full_run_name,
                )
                residual_trainer = pl.Trainer(
                    max_epochs=residual_epochs,
                    callbacks=residual_callbacks,
                    **trainer_args,
                )
                residual_trainer.fit(pcbm, train_dl, val_dl)
                _check_interruption(residual_trainer)
                _restore_checkpoint(
                    model=pcbm,
                    max_epochs=residual_epochs,
                    ckpt_call=residual_ckpt_call,
                    trainer=residual_trainer,
                )
                num_epochs.append(residual_trainer.current_epoch)
                times.append(time.time() - start_time)
                pcbm.unfreeze_non_residual_components()
                print(
                    f"\t\tDone with PCBM residual training "
                    f"after {residual_trainer.current_epoch} epochs ({times[-1]} sec)!"
                )


            if save_model:
                torch.save(
                    pcbm.state_dict(),
                    model_saved_path,
                )
                np.save(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([np.sum(times), np.sum(num_epochs)]),
                )
            training_time, num_epochs = np.sum(times), np.sum(num_epochs)
    eval_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
    )
    eval_results = evaluate.evaluate_cbm(
        model=pcbm,
        trainer=eval_trainer,
        config=config,
        run_name=run_name,
        old_results=old_results,
        rerun=rerun,
        test_dl=train_dl,
        dl_name="train",
    )
    eval_results['training_time'] = training_time
    eval_results['num_epochs'] = num_epochs
    eval_results[f'num_trainable_params'] = sum(
        p.numel() for p in pcbm.parameters() if p.requires_grad
    )
    eval_results[f'num_non_trainable_params'] = sum(
        p.numel() for p in pcbm.parameters() if not p.requires_grad
    )
    print(
        f'c_acc: {eval_results["train_acc_c"]*100:.2f}%, '
        f'y_acc: {eval_results["train_acc_y"]*100:.2f}%, '
        f'c_auc: {eval_results["train_auc_c"]*100:.2f}%, '
        f'y_auc: {eval_results["train_auc_y"]*100:.2f}% with '
        f'{num_epochs} epochs in {training_time:.2f} seconds'
    )
    return pcbm, eval_results