import copy
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

def train_mixcem(
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
            tmp_checkpoint_file = os.path.join(result_dir, f'{full_run_name}_tmp_checkpoint.pt')

            ####################################################################
            ## Step 0: Blackbox Warmup
            ####################################################################

            start_time = time.time()
            blackbox_warmup_epochs = config.get('blackbox_warmup_epochs', 0)
            if blackbox_warmup_epochs:
                print(
                    f"\tTraining entire model as a blackbox model for {blackbox_warmup_epochs} epochs"
                )
                warmup_callbacks, warmup_ckpt_call = _make_callbacks(
                    config,
                    result_dir,
                    full_run_name,
                )
                prev_task_loss_weight = model.task_loss_weight
                prev_concept_loss_weight = model.concept_loss_weight
                model.task_loss_weight = 1
                model.concept_loss_weight = 0
                model.warmup_mode = True
                warmup_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=blackbox_warmup_epochs,
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
                    max_epochs=blackbox_warmup_epochs,
                    ckpt_call=warmup_ckpt_call,
                    trainer=warmup_trainer,
                )
                training_time += time.time() - start_time
                num_epochs += warmup_trainer.current_epoch
                start_time = time.time()
                print(
                    f"\t\tDone after {num_epochs} epochs and {training_time} secs"
                )

                # Restore state
                model.task_loss_weight = prev_task_loss_weight
                model.concept_loss_weight = prev_concept_loss_weight
                model.warmup_mode = False


                # Now save to tmp checkpoint and reload model (in order to
                # restart any optimizer state)
                torch.save(
                    model.state_dict(),
                    tmp_checkpoint_file,
                )
                model = construct_model(
                    n_concepts,
                    n_tasks,
                    config,
                    imbalance=imbalance,
                    task_class_weights=task_class_weights,
                )
                model.load_state_dict(torch.load(tmp_checkpoint_file))

            ####################################################################
            ## Step 1: Train the concept model
            ####################################################################

            start_time = time.time()
            concept_epochs = config.get('concept_epochs', 0)
            if concept_epochs:
                print(
                    f"\tTraining up concept generator for {concept_epochs} epochs"
                )
                concept_callbacks, concept_ckpt_call = _make_callbacks(
                    config,
                    result_dir,
                    full_run_name,
                )
                prev_task_loss_weight = model.task_loss_weight
                prev_concept_loss_weight = model.concept_loss_weight
                prev_training_intervention_prob = model.training_intervention_prob
                model.task_loss_weight = 0
                model.concept_loss_weight = 1
                model.training_intervention_prob = 0
                concept_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=concept_epochs,
                    check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                    callbacks=concept_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )
                concept_trainer.fit(model, train_dl, val_dl)
                _check_interruption(concept_trainer)
                _restore_checkpoint(
                    model=model,
                    max_epochs=concept_epochs,
                    ckpt_call=concept_ckpt_call,
                    trainer=concept_trainer,
                )
                model.task_loss_weight = prev_task_loss_weight
                model.concept_loss_weight = prev_concept_loss_weight
                model.training_intervention_prob = prev_training_intervention_prob

                training_time += time.time() - start_time
                num_epochs += concept_trainer.current_epoch
                start_time = time.time()
                print(
                    f"\t\tDone after {num_epochs} epochs and {training_time} secs"
                )
                # Now save to tmp checkpoint and reload model (in order to
                # restart any optimizer state)
                torch.save(
                    model.state_dict(),
                    tmp_checkpoint_file,
                )
                model = construct_model(
                    n_concepts,
                    n_tasks,
                    config,
                    imbalance=imbalance,
                    task_class_weights=task_class_weights,
                )
                model.load_state_dict(torch.load(tmp_checkpoint_file))

            ####################################################################
            ## Step 2: Train the end-to-end model without any residual
            ####################################################################

            start_time = time.time()
            no_residual_epochs = config.get('no_residual_epochs', 0)
            if no_residual_epochs:
                print(
                    f"\tTraining end-to-end model WITHOUT residual for {no_residual_epochs} epochs"
                )
                # Setup
                model.freeze_residual()
                if config.get('fix_concept_embeddings_for_no_res', False):
                    model.freeze_concept_embeddings()
                if config.get('fix_backbone_for_no_res', True):
                    model.freeze_backbone()
                    if config.get('fix_concept_embeddings_for_no_res', False):
                        # Then there is no point in having a concept loss cause
                        # everything is frozen
                        prev_task_loss_weight = model.task_loss_weight
                        prev_concept_loss_weight = model.concept_loss_weight
                        model.task_loss_weight = 1
                        model.concept_loss_weight = 0

                # Training
                no_residual_callbacks, no_residual_ckpt_call = _make_callbacks(config, result_dir, full_run_name)
                no_residual_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=no_residual_epochs,
                    check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                    callbacks=no_residual_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )
                no_residual_trainer.fit(model, train_dl, val_dl)
                _check_interruption(no_residual_trainer)
                training_time += time.time() - start_time
                num_epochs += no_residual_trainer.current_epoch
                _restore_checkpoint(
                    model=model,
                    max_epochs=no_residual_epochs,
                    ckpt_call=no_residual_ckpt_call,
                    trainer=no_residual_trainer,
                )
                # And recover the state of the model
                model.unfreeze_residual()
                if config.get('fix_concept_embeddings_for_no_res', False):
                    model.unfreeze_concept_embeddings()
                if config.get('fix_backbone_for_no_res', True):
                    model.unfreeze_backbone()
                    if config.get('fix_concept_embeddings_for_no_res', False):
                        model.task_loss_weight = prev_task_loss_weight
                        model.concept_loss_weight = prev_concept_loss_weight

                # Now save to tmp checkpoint and reload model (in order to
                # restart any optimizer state)
                torch.save(
                    model.state_dict(),
                    tmp_checkpoint_file,
                )
                model = construct_model(
                    n_concepts,
                    n_tasks,
                    config,
                    imbalance=imbalance,
                    task_class_weights=task_class_weights,
                )
                model.load_state_dict(torch.load(tmp_checkpoint_file))


            ####################################################################
            ## Step 3: Train the end-to-end model with residual
            ####################################################################

            start_time = time.time()
            e2e_epochs = config.get('e2e_epochs', 0)
            if e2e_epochs:
                print(
                    f"\tTraining end-to-end model WITH residual for {e2e_epochs} epochs"
                )
                # Setup
                if config.get('fix_backbone_for_res', True):
                    model.freeze_backbone()
                if config.get('fix_concept_embeddings_for_res', False):
                    model.freeze_concept_embeddings()
                    if config.get('fix_concept_embeddings_for_res', False):
                        # Then there is no point in having a concept loss cause
                        # everything is frozen
                        prev_task_loss_weight = model.task_loss_weight
                        prev_concept_loss_weight = model.concept_loss_weight
                        model.task_loss_weight = 1
                        model.concept_loss_weight = 0
                if config.get('fix_label_predictor_for_res', False):
                    model.freeze_label_predictor()
                if config.get('use_ground_truth_mixing_for_res', False):
                    # Then we will make sure we use ground truth concepts when
                    # mixing embeddings
                    prev_training_intervention_prob = model.training_intervention_prob
                    model.training_intervention_prob = 1


                e2e_callbacks, e2e_ckpt_call = _make_callbacks(config, result_dir, full_run_name)
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

                # And undo any changes needed just for this stage
                if config.get('fix_backbone_for_res', True):
                    model.unfreeze_backbone()
                if config.get('fix_concept_embeddings_for_res', False):
                    model.unfreeze_concept_embeddings()
                    if config.get('fix_concept_embeddings_for_res', False):
                        model.task_loss_weight = prev_task_loss_weight
                        model.concept_loss_weight = prev_concept_loss_weight
                if config.get('fix_label_predictor_for_res', False):
                    model.unfreeze_label_predictor()
                if config.get('use_ground_truth_mixing_for_res', False):
                    # Then we will make sure we use ground truth concepts when
                    # mixing embeddings
                    model.training_intervention_prob = prev_training_intervention_prob


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