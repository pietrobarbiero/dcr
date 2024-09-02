
import copy
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import cem.models.standard as standard_models
import cem.train.evaluate as evaluate
import cem.train.utils as utils

from cem.train.training import _make_callbacks, _check_interruption, _restore_checkpoint


################################################################################
## Training Pipeline
################################################################################


def train_blackbox(
    input_shape,
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl,
    run_name,
    result_dir=None,
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
):
    enable_checkpointing = (
        True if config.get('early_stopping_best_model', False)
        else enable_checkpointing
    )
    assert activation_freq == 0, (
        'BlackBox training currently does not support activation dumping during '
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
        print("\tConstructing black-box model")
        architecture_config = config['architecture']
        bbox = standard_models.construct_standard_model(
            architecture=architecture_config['name'],
            input_shape=input_shape,
            n_labels=n_tasks,
            seed=seed,
            **architecture_config,
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
        if (not rerun) and os.path.exists(model_saved_path):
            # Then we simply load the model and proceed
            print("\t\tFound cached model... loading it")
            bbox.load_state_dict(torch.load(model_saved_path))
            if os.path.exists(
                model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [training_time, num_epochs] = np.load(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                )
            else:
                training_time, num_epochs = 0, 0
        else:
            bbox_epochs = config['max_epochs']
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
            _check_interruption(bbox)
            _restore_checkpoint(
                model=bbox,
                max_epochs=bbox_epochs,
                ckpt_call=bbox_ckpt_call,
                trainer=bbox_trainer,
            )
            num_epochs = bbox_trainer.current_epoch
            training_time = time.time() - start_time
            print(
                f"\t\tDone with black box model training "
                f"after {bbox_trainer.current_epoch} epochs ({training_time} secs)"
            )

            if save_model:
                joblib.dump(
                    config,
                    os.path.join(
                        result_dir,
                        f'{run_name}_experiment_config.joblib',
                    ),
                )
                torch.save(
                    bbox.state_dict(),
                    model_saved_path,
                )
                np.save(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([training_time, num_epochs]),
                )

    eval_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
    )
    eval_results = evaluate.evaluate_cbm(
        model=bbox,
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
        p.numel() for p in bbox.parameters() if p.requires_grad
    )
    eval_results[f'num_non_trainable_params'] = sum(
        p.numel() for p in bbox.parameters() if not p.requires_grad
    )
    print(
        f'c_acc: {eval_results["train_acc_c"]*100:.2f}%, '
        f'y_acc: {eval_results["train_acc_y"]*100:.2f}%, '
        f'c_auc: {eval_results["train_auc_c"]*100:.2f}%, '
        f'y_auc: {eval_results["train_auc_y"]*100:.2f}% with '
        f'{num_epochs} epochs in {training_time:.2f} seconds'
    )
    return bbox, eval_results