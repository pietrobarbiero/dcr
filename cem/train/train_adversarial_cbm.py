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

from cem.models.construction import construct_model

import cem.train.evaluate as evaluate
from cem.train.training import _make_callbacks, _check_interruption, _restore_checkpoint

################################################################################
## MODEL TRAINING
################################################################################

def train_adversarial_cbm(
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

    if (project_name) and result_dir and (
        not os.path.exists(os.path.join(result_dir, f'{full_run_name}.pt'))
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
        model_saved_path = os.path.join(
            result_dir or ".",
            f'{full_run_name}.pt'
        )
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
            training_time = 0
            num_epochs = 0

            ####################################################################
            ## Step 1: Hybrid CBM Warmup
            ####################################################################

            start_time = time.time()
            hybrid_cbm_epochs = config.get('hybrid_cbm_epochs', 0)
            if hybrid_cbm_epochs:
                print(
                    f"\tTraining Hybrid CBM for {hybrid_cbm_epochs} epochs"
                )
                if (not rerun) and "hybrid_path" in config and (
                    os.path.exists(os.path.join(
                        result_dir,
                        f'{config["hybrid_path"]}.pt'
                    ))
                ):
                    hybrid_cbm_saved_path = os.path.join(
                        result_dir,
                        f'{config["hybrid_path"]}.pt'
                    )
                    # Then we simply load the model and proceed
                    print("\tFound cached Hybrid CBM model... loading it")
                    model.load_state_dict(torch.load(hybrid_cbm_saved_path), strict=False)
                else:
                    hybrid_callbacks, hybrid_ckpt_call = _make_callbacks(
                        config,
                        result_dir,
                        full_run_name,
                    )
                    model.mode = 'cbm'
                    opt_configs = model.configure_optimizers()
                    hybrid_trainer = pl.Trainer(
                        accelerator=accelerator,
                        devices=devices,
                        max_epochs=hybrid_cbm_epochs,
                        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                        callbacks=hybrid_callbacks,
                        logger=logger or False,
                        enable_checkpointing=enable_checkpointing,
                        gradient_clip_val=gradient_clip_val,
                    )
                    hybrid_trainer.fit(model, train_dl, val_dl)
                    _check_interruption(hybrid_trainer)
                    _restore_checkpoint(
                        model=model,
                        max_epochs=hybrid_cbm_epochs,
                        ckpt_call=hybrid_ckpt_call,
                        trainer=hybrid_trainer,
                    )
                    training_time += time.time() - start_time
                    num_epochs += hybrid_trainer.current_epoch
                    start_time = time.time()
                    print(
                        f"\t\tDone after {num_epochs} epochs and {training_time} secs"
                    )

                    # Restart the optimizer state!
                    opts = model.optimizers()
                    opts.load_state_dict(opt_configs['optimizer'].state_dict())
                    if 'lr_scheduler' in opt_configs:
                        lr_scheduler = model.lr_schedulers()
                        lr_scheduler.load_state_dict(opt_configs['lr_scheduler'].state_dict())
                        lr_scheduler._reset()

                    if "hybrid_model_path" in config:
                        hybrid_model_saved_path = os.path.join(
                            result_dir or ".",
                            f'{config["hybrid_model_path"]}.pt'
                        )
                        torch.save(
                            model.state_dict(),
                            hybrid_model_saved_path,
                        )

            ####################################################################
            ## Step 2: Warmup the Discriminator
            ####################################################################

            start_time = time.time()
            warmup_epochs = config.get('warmup_epochs', 0)
            model.warmup_mode = False
            if warmup_epochs:
                print(
                    f"\tWarming up distriminator for {warmup_epochs} epochs"
                )

                # Else it is time to train it
                if (not rerun) and "warmup_model_path" in config and (
                    os.path.exists(os.path.join(
                        result_dir,
                        f'{config["warmup_model_path"]}.pt'
                    ))
                ):
                    warmup_model_saved_path = os.path.join(
                        result_dir or ".",
                        f'{config["warmup_model_path"]}.pt'
                    )
                    # Then we simply load the model and proceed
                    print("\tFound cached warmed-up model... loading it")
                    model.load_state_dict(torch.load(warmup_model_saved_path), strict=False)
                else:
                    warmup_callbacks, warmup_ckpt_call = _make_callbacks(
                        config,
                        result_dir,
                        full_run_name,
                    )
                    model.mode = 'discriminator_warmup'
                    opt_configs = model.configure_optimizers()
                    warmup_trainer = pl.Trainer(
                        accelerator=accelerator,
                        devices=devices,
                        max_epochs=warmup_epochs,
                        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                        callbacks=warmup_callbacks,
                        logger=logger or False,
                        enable_checkpointing=enable_checkpointing,
                        gradient_clip_val=gradient_clip_val,
                    )
                    warmup_trainer.fit(model, train_dl, val_dl)
                    _check_interruption(warmup_trainer)
                    _restore_checkpoint(
                        model=model,
                        max_epochs=warmup_epochs,
                        ckpt_call=warmup_ckpt_call,
                        trainer=warmup_trainer,
                    )
                    # Restart the optimizer state!
                    opts = model.optimizers()
                    opts.load_state_dict(opt_configs['optimizer'].state_dict())
                    if 'lr_scheduler' in opt_configs:
                        lr_scheduler = model.lr_schedulers()
                        lr_scheduler.load_state_dict(opt_configs['lr_scheduler'].state_dict())
                        lr_scheduler._reset()

                    training_time += time.time() - start_time
                    num_epochs += warmup_trainer.current_epoch
                    start_time = time.time()
                    print(
                        f"\t\tDone after {num_epochs} epochs and {training_time} secs"
                    )

                    if "warmup_model_path" in config:
                        warmup_model_saved_path = os.path.join(
                            result_dir or ".",
                            f'{config["warmup_model_path"]}.pt'
                        )
                        torch.save(
                            model.state_dict(),
                            warmup_model_saved_path,
                        )

            ####################################################################
            ## Step 3: Train the end-to-end model with adversarial loss
            ####################################################################

            start_time = time.time()
            e2e_epochs = config.get('e2e_epochs', 0)
            model.mode = 'cbm'
            if e2e_epochs:
                model.mode = 'joint'
                print(
                    f"\tTraining end-to-end model WITH adversarial loss for {e2e_epochs} epochs"
                )
                e2e_callbacks, e2e_ckpt_call = _make_callbacks(config, result_dir, full_run_name)
                opt_configs = model.configure_optimizers()
                e2e_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=e2e_epochs,
                    check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                    callbacks=e2e_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )
                e2e_trainer.fit(model, train_dl, val_dl)
                _check_interruption(e2e_trainer)
                training_time += time.time() - start_time
                num_epochs += e2e_trainer.current_epoch
                _restore_checkpoint(
                    model=model,
                    max_epochs=e2e_epochs,
                    ckpt_call=e2e_ckpt_call,
                    trainer=e2e_trainer,
                )

            if save_model and (result_dir is not None):
                torch.save(
                    model.state_dict(),
                    model_saved_path,
                )
                np.save(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([training_time, num_epochs]),
                )

        model.mode = 'cbm'

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