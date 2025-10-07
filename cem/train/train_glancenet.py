import copy
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything

import cem.train.evaluate as evaluate
import cem.train.utils as utils

from cem.models.construction import construct_model
from cem.models.glancenet import update_osr_thresholds
from cem.train.training import _make_callbacks, _check_interruption, _restore_checkpoint

def train_glancenet(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl=None,
    input_shape=None,
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
        run_name = "GlanceNet"

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
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        config=config,
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
        callbacks, ckpt_call = _make_callbacks(config, result_dir, full_run_name)
        fit_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=config['max_epochs'],
            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
            callbacks=callbacks,
            logger=logger or False,
            enable_checkpointing=enable_checkpointing,
            gradient_clip_val=gradient_clip_val,
        )

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
            # Else it is time to train it
            start_time = time.time()
            training_time = 0
            num_epochs = 0
            warmup_epochs = config.get('concept_warmup_epochs', 0)
            if warmup_epochs:
                print(
                    f"\tWarming up concept generator for {warmup_epochs} epochs"
                )
                warmup_callbacks, warmup_ckpt_call = _make_callbacks(
                    config,
                    result_dir,
                    full_run_name,
                )
                prev_task_loss_weight = model.task_loss_weight
                prev_concept_loss_weight = model.concept_loss_weight
                model.task_loss_weight = 0
                model.concept_loss_weight = 1
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
                if val_dl is not None:
                    warmup_trainer.fit(model, train_dl, val_dl)
                else:
                    warmup_trainer.fit(model, train_dl)
                _check_interruption(warmup_trainer)
                _restore_checkpoint(
                    model=model,
                    max_epochs=warmup_epochs,
                    ckpt_call=warmup_ckpt_call,
                    trainer=warmup_trainer,
                )
                model.task_loss_weight = prev_task_loss_weight
                model.concept_loss_weight = prev_concept_loss_weight
                training_time += time.time() - start_time
                num_epochs += warmup_trainer.current_epoch
                start_time = time.time()
                print(
                    f"\t\tDone after {num_epochs} epochs and {training_time} secs"
                )

            print(
                f"\tTraining end-to-end model for {config['max_epochs']} epochs"
            )
            if val_dl is not None:
                fit_trainer.fit(model, train_dl, val_dl)
            else:
                fit_trainer.fit(model, train_dl)
            _check_interruption(fit_trainer)
            training_time += time.time() - start_time
            num_epochs += fit_trainer.current_epoch
            _restore_checkpoint(
                model=model,
                max_epochs=config['max_epochs'],
                ckpt_call=ckpt_call,
                trainer=fit_trainer,
            )

            # Now time to compute the OSR thresholds
            model._osr_variables_set = False
            update_osr_thresholds(
                model=model,
                dataloader=train_dl,
                device=model.device,
                rec_percentile=config.get('rec_percentile', 95),
                proto_percentile=config.get('proto_percentile', 95),
            )
            model._osr_variables_set = True
            if save_model and (result_dir is not None):
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
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
        )
        eval_results = evaluate.evaluate_cbm(
            model=model,
            trainer=trainer,
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