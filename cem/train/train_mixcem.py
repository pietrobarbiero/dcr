import copy
import gc
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything

import cem.train.utils as utils
import cem.utils.data as data_utils

from cem.models.construction import construct_model

import cem.train.evaluate as evaluate
from cem.train.training import (
    _make_callbacks, _check_interruption, _restore_checkpoint
)


################################################################################
## MODEL TRAINING
################################################################################

def train_mixcem(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl=None,
    run_name=None,
    input_shape=None,
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
        run_name = "MixCEM"

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
        train=True,
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
    if config.get('load_weights_from'):
        load_file_name = os.path.join(
            result_dir,
            f'{config["load_weights_from"]}_fold_{split + 1}.pt',
        )
    else:
        load_file_name = os.path.join(result_dir, f'{full_run_name}.pt')
    model_saved_path = os.path.join(
        result_dir or ".",
        f'{full_run_name}.pt'
    )
    if (project_name) and result_dir and (
        not os.path.exists(load_file_name)
    ):
        # Lazy import to avoid importing unless necessary
        import wandb
        enter_obj = wandb.init(
            project=project_name,
            name=full_run_name,
            config=config,
            reinit=True
        )
    else:
        enter_obj = utils.EmptyEnter()

    with enter_obj as run:
        # Else it is time to train it
        if (not rerun) and os.path.exists(load_file_name):
            # Then we simply load the model and proceed
            print("\tFound cached model... loading it")
            model.load_state_dict(torch.load(load_file_name))
            if os.path.exists(
                load_file_name.replace(".pt", "_training_times.npy")
            ):
                [training_time, num_epochs] = np.load(
                    load_file_name.replace(".pt", "_training_times.npy"),
                )
            else:
                training_time, num_epochs = 0, 0
        else:
            training_time = 0
            num_epochs = 0
            load_path_name = config.get('load_path_name', None)
            loaded_weights = False
            if load_path_name:
                load_path_name = os.path.join(
                    result_dir,
                    load_path_name + f"_fold_{split + 1}.pt",
                )
            if load_path_name and os.path.exists(load_path_name):
                # Then we simply load the model and proceed
                print(
                    "\tFound model weights to load from for the initial "
                    "finetuning!"
                )
                model.load_state_dict(torch.load(load_path_name))
                loaded_weights = True


            ####################################################################
            ## Step 1: Train the end-to-end-model
            ####################################################################

            start_time = time.time()
            max_epochs = config.get('max_epochs', 100)
            if (not loaded_weights) and max_epochs:
                print(
                    f"\tTraining up the end-to-end model for {max_epochs} "
                    f"epochs"
                )

                # Else it is time to train it
                if (not rerun) and "end_to_end_model_path" in config and (
                    os.path.exists(os.path.join(
                        result_dir,
                        f'{config["end_to_end_model_path"]}.pt'
                    ))
                ):
                    e2e_model_saved_path = os.path.join(
                        result_dir or ".",
                        f'{config["end_to_end_model_path"]}.pt'
                    )
                    # Then we simply load the model and proceed
                    print("\tFound cached concept model... loading it")
                    state_dict = torch.load(e2e_model_saved_path)
                    model.load_state_dict(state_dict, strict=False)
                else:
                    print(
                        "[Number of parameters in model",
                        sum(
                            p.numel() for p in model.parameters()
                            if p.requires_grad
                        ),
                        "]"
                    )
                    print(
                        "[Number of non-trainable parameters in model",
                        sum(
                            p.numel() for p in model.parameters()
                            if not p.requires_grad
                        ),
                        "]",
                    )
                    e2e_callbacks, e2e_ckpt_call = _make_callbacks(
                        config,
                        result_dir,
                        full_run_name,
                    )
                    opt_configs = model.configure_optimizers()
                    e2e_trainer = pl.Trainer(
                        accelerator=accelerator,
                        devices=devices,
                        max_epochs=max_epochs,
                        check_val_every_n_epoch=config.get(
                            "check_val_every_n_epoch",
                            5,
                        ),
                        callbacks=e2e_callbacks,
                        logger=logger or False,
                        enable_checkpointing=enable_checkpointing,
                        gradient_clip_val=gradient_clip_val,
                    )
                    if val_dl is not None:
                        e2e_trainer.fit(model, train_dl, val_dl)
                    else:
                        e2e_trainer.fit(model, train_dl)
                    _check_interruption(e2e_trainer)
                    _restore_checkpoint(
                        model=model,
                        max_epochs=max_epochs,
                        ckpt_call=e2e_ckpt_call,
                        trainer=e2e_trainer,
                    )
                    # Restart the optimizer state!
                    opts = model.optimizers()
                    opts.load_state_dict(opt_configs['optimizer'].state_dict())
                    if 'lr_scheduler' in opt_configs:
                        lr_scheduler = model.lr_schedulers()
                        lr_scheduler.load_state_dict(
                            opt_configs['lr_scheduler'].state_dict()
                        )
                        lr_scheduler._reset()

                    training_time += time.time() - start_time
                    num_epochs += e2e_trainer.current_epoch
                    start_time = time.time()
                    print(
                        f"\t\tDone after {num_epochs} epochs "
                        f"and {training_time} secs"
                    )

                    if "end_to_end_model_path" in config:
                        e2e_model_saved_path = os.path.join(
                            result_dir or ".",
                            f'{config["end_to_end_model_path"]}.pt'
                        )
                        torch.save(
                            model.state_dict(),
                            e2e_model_saved_path,
                        )

            if config.get('freeze_global_components', False):
                model.unfreeze_global_components()

            eval_trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                logger=False,
            )

            ####################################################################
            ## Step 2: Calibrate logits (DONE by default)
            ####################################################################

            start_time = time.time()
            calibration_epochs = config.get('calibration_epochs', 30)
            include_calibration = not (loaded_weights and config.get(
                'no_calibration_if_found',
                False,
            ))
            if include_calibration and calibration_epochs:
                print(
                    f"\tCalibrating model for {calibration_epochs} epochs"
                )
                trainable_params = set()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        trainable_params.add(name)
                        param.requires_grad = False
                model.unfreeze_calibration_components(
                    unfreeze_dynamic=config.get(
                        'calibrate_dynamic_logits',
                        True,
                    ),
                    unfreeze_global=config.get(
                        'calibrate_global_logits',
                        True,
                    ),
                )

                if config.get('ignore_task_acc_in_calibration', False):
                    prev_task_loss_weight = model.task_loss_weight
                    model.task_loss_weight = 0
                    prev_all_intervened_loss_weight = model.all_intervened_loss_weight
                    model.all_intervened_loss_weight = 0

                calibration_callbacks, calibration_ckpt_call = \
                    _make_callbacks(
                        config,
                        result_dir,
                        full_run_name,
                    )
                print(
                    "[Number of parameters in model",
                    sum(
                        p.numel() for p in model.parameters()
                        if p.requires_grad
                    ),
                    "]"
                )
                print(
                    "[Number of non-trainable parameters in model",
                    sum(
                        p.numel() for p in model.parameters()
                        if not p.requires_grad
                    ),
                    "]",
                )
                opt_configs = model.configure_optimizers()
                calibration_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=calibration_epochs,
                    check_val_every_n_epoch=config.get(
                        "check_val_every_n_epoch",
                        5,
                    ),
                    callbacks=calibration_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )

                # Train it on the validation set!
                if config.get('finetune_with_val', True) and (
                    val_dl is not None
                ):
                    calibration_trainer.fit(model, val_dl, val_dl)
                else:
                    calibration_trainer.fit(model, train_dl)
                _check_interruption(calibration_trainer)
                training_time += time.time() - start_time
                num_epochs += calibration_trainer.current_epoch
                _restore_checkpoint(
                    model=model,
                    max_epochs=calibration_epochs,
                    ckpt_call=calibration_ckpt_call,
                    trainer=calibration_trainer,
                )
                if config.get('ignore_task_acc_in_calibration', False):
                    model.task_loss_weight = prev_task_loss_weight
                    model.all_intervened_loss_weight = \
                        prev_all_intervened_loss_weight

                # And restore the state
                model.hard_selection_value = None
                for name, param in model.named_parameters():
                    param.requires_grad = name in trainable_params


        if save_model and (result_dir is not None) and (
            not os.path.exists(model_saved_path)
        ):
            torch.save(
                model.state_dict(),
                model_saved_path,
            )
            np.save(
                model_saved_path.replace(".pt", "_training_times.npy"),
                np.array([training_time, num_epochs]),
            )


        if not os.path.exists(os.path.join(
            result_dir,
            f'{run_name}_experiment_config.joblib'
        )):
            # Then let's serialize the experiment config for this run
            config_copy = copy.deepcopy(config)
            if "c_extractor_arch" in config_copy and (
                not isinstance(config_copy["c_extractor_arch"], str)
            ):
                del config_copy["c_extractor_arch"]
            joblib.dump(config_copy, os.path.join(
                result_dir,
                f'{run_name}_experiment_config.joblib'
            ))

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
