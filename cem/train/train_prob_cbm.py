import copy
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import cem.train.evaluate as evaluate
import cem.train.utils as utils

from cem.models.construction import construct_model
from cem.train.training import (
    _make_callbacks, _check_interruption
)

def train_prob_cbm(
    input_shape,
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl=None,
    run_name=None,
    result_dir=None,
    split=None,
    imbalance=None,
    task_class_weights=None,
    rerun=False,
    logger=False,
    project_name='',
    seed=None,
    save_model=True,
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
    if seed is not None:
        seed_everything(seed)

    if run_name is None:
        run_name = "ProbCBM"

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

    # create model
    model = construct_model(
        n_concepts,
        n_tasks,
        config,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
    )
    print(
        "[Number of parameters in model",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        "]"
    )
    print(
        "[Number of non-trainable parameters in model",
        sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "]",
    )
    if config.get("model_pretrain_path"):
        if os.path.exists(config.get("model_pretrain_path")):
            # Then we simply load the model and proceed
            print("\tFound pretrained model to load the initial weights from!")
            model.load_state_dict(
                torch.load(config.get("model_pretrain_path")),
                strict=False,
            )

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

    callbacks, ckpt_call = _make_callbacks(config, result_dir, full_run_name)
    trainer_args = dict(
        accelerator=accelerator,
        devices=devices,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        gradient_clip_val=gradient_clip_val,
        # Only use the wandb logger when it is a fresh run
        logger=used_logger,
    )

    with enter_obj as run:
        if (not rerun) and os.path.exists(model_saved_path):
            # Then we simply load the model and proceed
            print("\tFound cached model... loading it")
            model.load_state_dict(torch.load(model_saved_path))
            if os.path.exists(
                model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [training_time, num_epochs] = np.load(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                )
            else:
                training_time, num_epochs = 0, 0
        else:
            # Else it is time to train it
            start_time = time.time()
            # First train the concept model by setting the mode accordingly
            if model.train_class_mode == 'sequential':
                trainable_params = [
                    name for name, param in model.named_parameters()
                    if param.requires_grad
                ]
                old_lrs = [
                    config['learning_rate'],
                    config['learning_rate'] * config.get('lr_ratio', 10),
                ]
                if config.get('warmup_epochs', 5) != 0:
                    warmup_epochs = config.get('warmup_epochs', 5)
                    print(
                        f"\tWarming up ProbCBM for {warmup_epochs} epochs"
                    )
                    for p in model.extractor.parameters():
                        p.requires_grad = False
                    warmup_trainer = pl.Trainer(
                        max_epochs=config.get(
                            'warmup_epochs',
                            5,
                        ),
                        **trainer_args,
                    )
                    if val_dl is not None:
                        warmup_trainer.fit(model, train_dl, val_dl)
                    else:
                        warmup_trainer.fit(model, train_dl)
                    _check_interruption(warmup_trainer)
                    for p in model.extractor.parameters():
                        p.requires_grad = True
                    print("\t\tDone with warmup!")
                print("\tTraining ProbCBM's concept model")
                model.stage = 'concept'
                params_to_train = [
                    name for name, _ in model.named_parameters()
                    if name not in model.params_to_classify()
                ]
                # Make sure we unfreeze only the correct weights
                for name, parameter in model.named_parameters():
                    if name not in params_to_train:
                        parameter.requires_grad = False
                    elif name in trainable_params:
                        parameter.requires_grad = True
                max_epochs = config.get(
                        'max_concept_epochs',
                        config.get('max_epochs', None),
                    ) - config.get('warmup_epochs', 5)
                concept_trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    **trainer_args,
                )
                if val_dl is not None:
                    concept_trainer.fit(model, train_dl, val_dl)
                else:
                    concept_trainer.fit(model, train_dl)
                _check_interruption(concept_trainer)
                num_epochs = concept_trainer.current_epoch
                if config.get('early_stopping_best_model', False) and (
                    concept_trainer.current_epoch != max_epochs
                ):
                    # Then restore the best validation model
                    chkpoint = torch.load(ckpt_call.best_model_path)
                    model.load_state_dict(chkpoint["state_dict"])
                print("\tTraining ProbCBM's task model")
                model.stage = 'class'
                params_to_train = [
                    name for name, _ in model.named_parameters()
                    if name in model.params_to_classify()
                ]
                # Make sure we unfreeze only the correct weights
                for name, parameter in model.named_parameters():
                    if name not in params_to_train:
                        parameter.requires_grad = False
                    elif name in trainable_params:
                        parameter.requires_grad = True
                # Reset learning rates too
                for g, old_lr in zip(
                    model.optimizers().param_groups,
                    old_lrs,
                ):
                    g['lr'] = old_lr
                max_epochs = config.get(
                    'max_task_epochs',
                    config.get('max_epochs', None),
                )
                task_trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    **trainer_args,
                )
                if val_dl is not None:
                    task_trainer.fit(model, train_dl, val_dl)
                else:
                    task_trainer.fit(model, train_dl)
                _check_interruption(task_trainer)
                num_epochs += task_trainer.current_epoch
                if config.get('early_stopping_best_model', False) and (
                    task_trainer.current_epoch != max_epochs
                ):
                    # Then restore the best validation model
                    chkpoint = torch.load(ckpt_call.best_model_path)
                    model.load_state_dict(chkpoint["state_dict"])
                training_time = time.time() - start_time
            elif model.train_class_mode == 'joint':
                print("\tTraining ProbCBM jointly")
                max_epochs = config['max_epochs']
                task_trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    **trainer_args,
                )
                model.stage = 'joint'
                if val_dl is not None:
                    task_trainer.fit(model, train_dl, val_dl)
                else:
                    task_trainer.fit(model, train_dl)
                _check_interruption(task_trainer)
                num_epochs = task_trainer.current_epoch
                if config.get('early_stopping_best_model', False) and (
                    task_trainer.current_epoch != max_epochs
                ):
                    # Then restore the best validation model
                    chkpoint = torch.load(
                        "ckpt_call.best_model_path =",
                        ckpt_call.best_model_path,
                    )
                    model.load_state_dict(chkpoint["state_dict"])
                training_time = time.time() - start_time
            else:
                raise ValueError(
                    f'Currently we only support sequential or jointly '
                    f'trained ProbCBMs. We do not support '
                    f'train_class_mode = {model.train_class_mode}.'
                )
            config_copy = copy.deepcopy(config)
            if "c_extractor_arch" in config_copy and (
                not isinstance(config_copy["c_extractor_arch"], str)
            ):
                del config_copy["c_extractor_arch"]
            joblib.dump(
                config_copy,
                os.path.join(
                    result_dir,
                    f'{run_name}_experiment_config.joblib',
                ),
            )
            if save_model:
                torch.save(
                    model.state_dict(),
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
        model=model,
        trainer=eval_trainer,
        config=config,
        run_name=run_name,
        old_results=old_results,
        rerun=rerun,
        test_dl=train_dl, # Evaluate training metrics
        dl_name="train",
    )
    eval_results['training_time'] = training_time
    eval_results['num_epochs'] = num_epochs
    eval_results[f'num_trainable_params'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    eval_results[f'num_non_trainable_params'] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    print(
        f'c_acc: {eval_results["train_acc_c"]*100:.2f}%, '
        f'y_acc: {eval_results["train_acc_y"]*100:.2f}%, '
        f'c_auc: {eval_results["train_auc_c"]*100:.2f}%, '
        f'y_auc: {eval_results["train_auc_y"]*100:.2f}% with '
        f'{num_epochs} epochs in {training_time:.2f} seconds'
    )
    return model, eval_results
