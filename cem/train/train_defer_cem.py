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

def train_defer_cem(
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

            ####################################################################
            ## Step 1: Train the concept model
            ####################################################################

            start_time = time.time()
            concept_epochs = config.get('mixcem_concept_epochs', 0)
            if concept_epochs:
                model.mode = 'mixcem'
                print(
                    f"\tTraining up MixCEM concept generator for {concept_epochs} epochs"
                )

                # Else it is time to train it
                if (not rerun) and "mixcem_concept_model_path" in config and (
                    os.path.exists(os.path.join(
                        result_dir,
                        f'{config["mixcem_concept_model_path"]}.pt'
                    ))
                ):
                    concept_model_saved_path = os.path.join(
                        result_dir or ".",
                        f'{config["mixcem_concept_model_path"]}.pt'
                    )
                    # Then we simply load the model and proceed
                    print("\tFound cached concept model... loading it")
                    state_dict = torch.load(concept_model_saved_path)
                    model.load_state_dict(state_dict, strict=False)
                else:
                    concept_callbacks, concept_ckpt_call = _make_callbacks(
                        config,
                        result_dir,
                        full_run_name,
                    )
                    prev_task_loss_weight = model.task_loss_weight
                    prev_concept_loss_weight = model.concept_loss_weight
                    prev_num_rollouts = model.num_rollouts
                    prev_training_intervention_prob = model.training_intervention_prob
                    model.num_rollouts = 0
                    model.task_loss_weight = 0
                    model.concept_loss_weight = 1
                    model.training_intervention_prob = 0
                    opt_configs = model.configure_optimizers()
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
                    # Restart the optimizer state!
                    opts = model.optimizers()
                    opts.load_state_dict(opt_configs['optimizer'].state_dict())
                    if 'lr_scheduler' in opt_configs:
                        lr_scheduler = model.lr_schedulers()
                        lr_scheduler.load_state_dict(opt_configs['lr_scheduler'].state_dict())
                        lr_scheduler._reset()

                    model.task_loss_weight = prev_task_loss_weight
                    model.concept_loss_weight = prev_concept_loss_weight
                    model.training_intervention_prob = prev_training_intervention_prob
                    model.num_rollouts = prev_num_rollouts
                    if config.get('fix_backbone_for_concept', False):
                        model.unfreeze_backbone()

                    training_time += time.time() - start_time
                    num_epochs += concept_trainer.current_epoch
                    start_time = time.time()
                    print(
                        f"\t\tDone after {num_epochs} epochs and {training_time} secs"
                    )

                    if "mixcem_concept_model_path" in config:
                        concept_model_saved_path = os.path.join(
                            result_dir or ".",
                            f'{config["mixcem_concept_model_path"]}.pt'
                        )
                        torch.save(
                            model.state_dict(),
                            concept_model_saved_path,
                        )
            if config.get('first_dynamic', False):
                ####################################################################
                ## Step 2: Train the dynamic CEM model
                ####################################################################

                start_time = time.time()
                dynamic_epochs = config.get('dynamic_epochs', 0)
                if dynamic_epochs:
                    model.mode = 'dynamic'
                    print(
                        f"\tTraining (dynamic) CEM for {dynamic_epochs} epochs"
                    )
                    if (not rerun) and "dynamic_entire_model_path" in config and (
                        os.path.exists(os.path.join(
                            result_dir,
                            f'{config["dynamic_entire_model_path"]}.pt'
                        ))
                    ):
                        concept_model_saved_path = os.path.join(
                            result_dir or ".",
                            f'{config["dynamic_entire_model_path"]}.pt'
                        )
                        # Then we simply load the model and proceed
                        print("\tFound cached concept model... loading it")
                        state_dict = torch.load(concept_model_saved_path)
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        # Setup
                        model.freeze_mixcem_model()
                        model.freeze_pred_mixture_model()
                        if config.get('fix_backbone_for_dynamic', False):
                            model.freeze_backbone(freeze_emb_generators=config.get('freeze_emb_generators_for_dynamic', True))

                        dynamic_callbacks, dynamic_ckpt_call = _make_callbacks(config, result_dir, full_run_name)
                        opt_configs = model.configure_optimizers()
                        dynamic_trainer = pl.Trainer(
                            accelerator=accelerator,
                            devices=devices,
                            max_epochs=dynamic_epochs,
                            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                            callbacks=dynamic_callbacks,
                            logger=logger or False,
                            enable_checkpointing=enable_checkpointing,
                            gradient_clip_val=gradient_clip_val,
                        )
                        dynamic_trainer.fit(model, train_dl, val_dl)
                        _check_interruption(dynamic_trainer)
                        training_time += time.time() - start_time
                        num_epochs += dynamic_trainer.current_epoch
                        _restore_checkpoint(
                            model=model,
                            max_epochs=dynamic_epochs,
                            ckpt_call=dynamic_ckpt_call,
                            trainer=dynamic_trainer,
                        )

                        # And undo any changes needed just for this stage
                        model.unfreeze_mixcem_model()
                        if config.get('fix_backbone_for_dynamic', False):
                            model.unfreeze_backbone()
                        model.unfreeze_pred_mixture_model()

                        if "dynamic_entire_model_path" in config:
                            concept_model_saved_path = os.path.join(
                                result_dir or ".",
                                f'{config["dynamic_entire_model_path"]}.pt'
                            )
                            torch.save(
                                model.state_dict(),
                                concept_model_saved_path,
                            )

                ####################################################################
                ## Step 3: Train Mixcem (global embeddings)
                ####################################################################

                start_time = time.time()
                mixcem_epochs = config.get('mixcem_epochs', 0)
                if mixcem_epochs:
                    model.mode = 'mixcem'
                    print(
                        f"\tTraining end-to-end MixCEM global components for {mixcem_epochs} epochs"
                    )
                    if (not rerun) and "mixcem_entire_model_path" in config and (
                        os.path.exists(os.path.join(
                            result_dir,
                            f'{config["mixcem_entire_model_path"]}.pt'
                        ))
                    ):
                        concept_model_saved_path = os.path.join(
                            result_dir or ".",
                            f'{config["mixcem_entire_model_path"]}.pt'
                        )
                        # Then we simply load the model and proceed
                        print("\tFound cached concept model... loading it")
                        state_dict = torch.load(concept_model_saved_path)
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        # Setup
                        model.freeze_dynamic_model()
                        model.freeze_pred_mixture_model()
                        if config.get('fix_concept_embeddings_for_mixcem', False):
                            model.freeze_concept_embeddings()
                        if config.get('fix_backbone_for_mixcem', False):
                            model.freeze_backbone(freeze_emb_generators=config.get('freeze_emb_generators_for_mixcem', True))
                            if config.get('fix_concept_embeddings_for_mixcem', False) and (
                                config.get('freeze_emb_generators_for_mixcem', True)
                            ):
                                # Then there is no point in having a concept loss cause
                                # everything is frozen
                                prev_task_loss_weight = model.task_loss_weight
                                prev_concept_loss_weight = model.concept_loss_weight
                                model.task_loss_weight = 1
                                model.concept_loss_weight = 0

                        # Training
                        mixcem_callbacks, mixcem_ckpt_call = _make_callbacks(config, result_dir, full_run_name)
                        opt_configs = model.configure_optimizers()
                        mixcem_trainer = pl.Trainer(
                            accelerator=accelerator,
                            devices=devices,
                            max_epochs=mixcem_epochs,
                            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                            callbacks=mixcem_callbacks,
                            logger=logger or False,
                            enable_checkpointing=enable_checkpointing,
                            gradient_clip_val=gradient_clip_val,
                        )
                        mixcem_trainer.fit(model, train_dl, val_dl)
                        _check_interruption(mixcem_trainer)
                        training_time += time.time() - start_time
                        num_epochs += mixcem_trainer.current_epoch
                        _restore_checkpoint(
                            model=model,
                            max_epochs=mixcem_epochs,
                            ckpt_call=mixcem_ckpt_call,
                            trainer=mixcem_trainer,
                        )
                        # Restart the optimizer state!
                        opts = model.optimizers()
                        opts.load_state_dict(opt_configs['optimizer'].state_dict())
                        if 'lr_scheduler' in opt_configs:
                            lr_scheduler = model.lr_schedulers()
                            lr_scheduler.load_state_dict(opt_configs['lr_scheduler'].state_dict())
                            lr_scheduler._reset()

                        # And recover the state of the model
                        model.unfreeze_dynamic_model()
                        if config.get('fix_concept_embeddings_for_mixcem', False):
                            model.unfreeze_concept_embeddings()
                        if config.get('fix_backbone_for_mixcem', False):
                            model.unfreeze_backbone()
                            if config.get('fix_concept_embeddings_for_mixcem', False) and (
                                config.get('freeze_emb_generators_for_mixcem', True)
                            ):
                                model.task_loss_weight = prev_task_loss_weight
                                model.concept_loss_weight = prev_concept_loss_weight
                        model.unfreeze_pred_mixture_model()

                        if "mixcem_entire_model_path" in config:
                            concept_model_saved_path = os.path.join(
                                result_dir or ".",
                                f'{config["mixcem_entire_model_path"]}.pt'
                            )
                            torch.save(
                                model.state_dict(),
                                concept_model_saved_path,
                            )

            else:

                ####################################################################
                ## Step 2: Train Mixcem (global embeddings)
                ####################################################################

                start_time = time.time()
                mixcem_epochs = config.get('mixcem_epochs', 0)
                if mixcem_epochs:
                    model.mode = 'mixcem'
                    print(
                        f"\tTraining end-to-end MixCEM global components for {mixcem_epochs} epochs"
                    )
                    if (not rerun) and "mixcem_entire_model_path" in config and (
                        os.path.exists(os.path.join(
                            result_dir,
                            f'{config["mixcem_entire_model_path"]}.pt'
                        ))
                    ):
                        concept_model_saved_path = os.path.join(
                            result_dir or ".",
                            f'{config["mixcem_entire_model_path"]}.pt'
                        )
                        # Then we simply load the model and proceed
                        print("\tFound cached concept model... loading it")
                        state_dict = torch.load(concept_model_saved_path)
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        # Setup
                        model.freeze_dynamic_model()
                        model.freeze_pred_mixture_model()
                        if config.get('fix_concept_embeddings_for_mixcem', False):
                            model.freeze_concept_embeddings()
                        if config.get('fix_backbone_for_mixcem', False):
                            model.freeze_backbone(freeze_emb_generators=config.get('freeze_emb_generators_for_mixcem', True))
                            if config.get('fix_concept_embeddings_for_mixcem', False) and (
                                config.get('freeze_emb_generators_for_mixcem', True)
                            ):
                                # Then there is no point in having a concept loss cause
                                # everything is frozen
                                prev_task_loss_weight = model.task_loss_weight
                                prev_concept_loss_weight = model.concept_loss_weight
                                model.task_loss_weight = 1
                                model.concept_loss_weight = 0

                        # Training
                        mixcem_callbacks, mixcem_ckpt_call = _make_callbacks(config, result_dir, full_run_name)
                        opt_configs = model.configure_optimizers()
                        mixcem_trainer = pl.Trainer(
                            accelerator=accelerator,
                            devices=devices,
                            max_epochs=mixcem_epochs,
                            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                            callbacks=mixcem_callbacks,
                            logger=logger or False,
                            enable_checkpointing=enable_checkpointing,
                            gradient_clip_val=gradient_clip_val,
                        )
                        mixcem_trainer.fit(model, train_dl, val_dl)
                        _check_interruption(mixcem_trainer)
                        training_time += time.time() - start_time
                        num_epochs += mixcem_trainer.current_epoch
                        _restore_checkpoint(
                            model=model,
                            max_epochs=mixcem_epochs,
                            ckpt_call=mixcem_ckpt_call,
                            trainer=mixcem_trainer,
                        )
                        # Restart the optimizer state!
                        opts = model.optimizers()
                        opts.load_state_dict(opt_configs['optimizer'].state_dict())
                        if 'lr_scheduler' in opt_configs:
                            lr_scheduler = model.lr_schedulers()
                            lr_scheduler.load_state_dict(opt_configs['lr_scheduler'].state_dict())
                            lr_scheduler._reset()

                        # And recover the state of the model
                        model.unfreeze_dynamic_model()
                        if config.get('fix_concept_embeddings_for_mixcem', False):
                            model.unfreeze_concept_embeddings()
                        if config.get('fix_backbone_for_mixcem', False):
                            model.unfreeze_backbone()
                            if config.get('fix_concept_embeddings_for_mixcem', False) and (
                                config.get('freeze_emb_generators_for_mixcem', True)
                            ):
                                model.task_loss_weight = prev_task_loss_weight
                                model.concept_loss_weight = prev_concept_loss_weight
                        model.unfreeze_pred_mixture_model()

                        if "mixcem_entire_model_path" in config:
                            concept_model_saved_path = os.path.join(
                                result_dir or ".",
                                f'{config["mixcem_entire_model_path"]}.pt'
                            )
                            torch.save(
                                model.state_dict(),
                                concept_model_saved_path,
                            )




                ####################################################################
                ## Step 3: Train the dynamic CEM model
                ####################################################################

                start_time = time.time()
                dynamic_epochs = config.get('dynamic_epochs', 0)
                if dynamic_epochs:
                    model.mode = 'dynamic'
                    print(
                        f"\tTraining (dynamic) CEM for {dynamic_epochs} epochs"
                    )
                    if (not rerun) and "dynamic_entire_model_path" in config and (
                        os.path.exists(os.path.join(
                            result_dir,
                            f'{config["dynamic_entire_model_path"]}.pt'
                        ))
                    ):
                        concept_model_saved_path = os.path.join(
                            result_dir or ".",
                            f'{config["dynamic_entire_model_path"]}.pt'
                        )
                        # Then we simply load the model and proceed
                        print("\tFound cached concept model... loading it")
                        state_dict = torch.load(concept_model_saved_path)
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        # Setup
                        model.freeze_mixcem_model()
                        model.freeze_pred_mixture_model()
                        if config.get('fix_backbone_for_dynamic', False):
                            model.freeze_backbone(freeze_emb_generators=config.get('freeze_emb_generators_for_dynamic', True))

                        dynamic_callbacks, dynamic_ckpt_call = _make_callbacks(config, result_dir, full_run_name)
                        opt_configs = model.configure_optimizers()
                        dynamic_trainer = pl.Trainer(
                            accelerator=accelerator,
                            devices=devices,
                            max_epochs=dynamic_epochs,
                            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                            callbacks=dynamic_callbacks,
                            logger=logger or False,
                            enable_checkpointing=enable_checkpointing,
                            gradient_clip_val=gradient_clip_val,
                        )
                        dynamic_trainer.fit(model, train_dl, val_dl)
                        _check_interruption(dynamic_trainer)
                        training_time += time.time() - start_time
                        num_epochs += dynamic_trainer.current_epoch
                        _restore_checkpoint(
                            model=model,
                            max_epochs=dynamic_epochs,
                            ckpt_call=dynamic_ckpt_call,
                            trainer=dynamic_trainer,
                        )

                        # And undo any changes needed just for this stage
                        model.unfreeze_mixcem_model()
                        if config.get('fix_backbone_for_dynamic', False):
                            model.unfreeze_backbone()
                        model.unfreeze_pred_mixture_model()

                        if "dynamic_entire_model_path" in config:
                            concept_model_saved_path = os.path.join(
                                result_dir or ".",
                                f'{config["dynamic_entire_model_path"]}.pt'
                            )
                            torch.save(
                                model.state_dict(),
                                concept_model_saved_path,
                            )



            ####################################################################
            ## Step 4: Train the entire model end-to-end
            ####################################################################

            start_time = time.time()
            joint_epochs = config.get('joint_epochs', 0)
            if joint_epochs:
                model.mode = 'joint'
                print(
                    f"\tTraining end-to-end joint model for {joint_epochs} epochs"
                )
                # Setup
                if config.get('fix_backbone_for_joint', False):
                    model.freeze_backbone(freeze_emb_generators=config.get('freeze_emb_generators_for_joint', True))
                if config.get('fix_concept_embeddings_for_joint', False):
                    model.freeze_concept_embeddings()
                if config.get('fix_dynamic_prob_generators_for_joint', False):
                    model.freeze_dynamic_prob_generators()
                if config.get('fix_mixcem_label_predictor_for_joint', False):
                    model.freeze_mixcem_label_predictor()
                if config.get('fix_dynamic_label_predictor_for_joint', False):
                    model.freeze_dynamic_label_predictor()

                joint_callbacks, joint_ckpt_call = _make_callbacks(config, result_dir, full_run_name)
                opt_configs = model.configure_optimizers()
                joint_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=joint_epochs,
                    check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                    callbacks=joint_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )
                joint_trainer.fit(model, train_dl, val_dl)
                _check_interruption(joint_trainer)
                training_time += time.time() - start_time
                num_epochs += joint_trainer.current_epoch
                _restore_checkpoint(
                    model=model,
                    max_epochs=joint_epochs,
                    ckpt_call=joint_ckpt_call,
                    trainer=joint_trainer,
                )

                # And undo any changes needed just for this stage
                if config.get('fix_backbone_for_joint', False):
                    model.unfreeze_backbone()
                if config.get('fix_concept_embeddings_for_joint', False):
                    model.unfreeze_concept_embeddings()
                if config.get('fix_dynamic_prob_generators_for_joint', False):
                    model.unfreeze_dynamic_prob_generators()
                if config.get('fix_mixcem_label_predictor_for_joint', False):
                    model.unfreeze_mixcem_label_predictor()
                if config.get('fix_dynamic_label_predictor_for_joint', False):
                    model.unfreeze_dynamic_label_predictor()

            # And we are done!
            model.mode = 'joint'

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