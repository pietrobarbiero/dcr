import copy
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import cem.train.evaluate as evaluate
import cem.train.utils as utils
import cem.utils.data as data_utils

from cem.models.construction import construct_model
from cem.train.training import _make_callbacks, _check_interruption, _restore_checkpoint

def train_fixed_cem(
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
    initialized_concept_embs = False
    trained_already = False
    weights_path = None
    if config.get("weights_path"):
        weights_path = config.get("weights_path")
        if not os.path.exists(weights_path):
            weights_path = weights_path + f"_fold_{split + 1}.pt"
    if weights_path is not None:
        if os.path.exists(weights_path):
            # Then we simply load the model and proceed
            print("\tFound pretrained model to load the initial weights from!")
            model.load_state_dict(
                torch.load(weights_path),
                strict=False,
            )
            trained_already = True

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
        print("model_saved_path =", model_saved_path)

        if (not rerun) and os.path.exists(model_saved_path):
            # Then we simply load the model and proceed
            print("\tFound cached model... loading it")
            initialized_concept_embs = True
            model.load_state_dict(torch.load(model_saved_path))
            if os.path.exists(
                model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [training_time, num_epochs] = np.load(
                    model_saved_path.replace(".pt", "_training_times.npy"),
                )
            else:
                training_time, num_epochs = 0, 0
        elif trained_already:
            print("Model has been already trained, so we will not do more training.")
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
                warmup_trainer.fit(model, train_dl, val_dl)
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
            fit_trainer.fit(model, train_dl, val_dl)
            _check_interruption(fit_trainer)
            training_time += time.time() - start_time
            num_epochs += fit_trainer.current_epoch
            _restore_checkpoint(
                model=model,
                max_epochs=config['max_epochs'],
                ckpt_call=ckpt_call,
                trainer=fit_trainer,
            )

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            enable_checkpointing=enable_checkpointing,
        )
        if not initialized_concept_embs:
            # Now time to compute the static concept weights we will use in this
            model._set_embeddings = False
            print("\tInitializing fixed concept embeddings based on averages")
            model.output_embeddings = True
            batch_results = trainer.predict(model, train_dl)
            pos_embs = np.concatenate(
                list(map(lambda x: x[-2], batch_results)),
                axis=0,
            )
            neg_embs = np.concatenate(
                list(map(lambda x: x[-1], batch_results)),
                axis=0,
            )
            active_intervention_values = []
            inactive_intervention_values = []
            active_intervention_values = np.mean(
                pos_embs,
                axis=0,
            )
            inactive_intervention_values = np.mean(
                neg_embs,
                axis=0,
            )
            model.output_embeddings = False
            model._set_embeddings = True # Make sure these are set!
            with torch.no_grad():
                model.concept_embeddings.copy_(torch.tensor(
                    np.concatenate(
                        [
                            np.expand_dims(active_intervention_values, axis=1),
                            np.expand_dims(inactive_intervention_values, axis=1),
                        ],
                        axis=1,
                    )
                ))
        else:
            print(
                "Concept embeddings have already been initialized! So no need "
                "to do this again."
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