import copy
import gc
import joblib
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from sklearn.mixture import GaussianMixture
from pytorch_lightning import seed_everything
from scipy.stats import beta as beta_fn

import cem.train.utils as utils
import cem.utils.data as data_utils

from cem.models.construction import construct_model

import cem.train.evaluate as evaluate
from cem.train.training import _make_callbacks, _check_interruption, _restore_checkpoint




import torch.distributions as dist

class MixtureOfExperts(torch.nn.Module):
    def __init__(self, input_dim, num_components):
        super(MixtureOfExperts, self).__init__()
        # Gating network: takes input and outputs mixture weights
        self.input_dim = input_dim
        self.num_components = num_components
        self.gating_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_components),
            torch.nn.Softmax(dim=-1)
        )
        # Define the Gaussian components (means and covariances)
        self.means = torch.nn.Parameter(torch.randn(num_components, input_dim))
        # Diagonal covariance parameters (before applying exp to ensure positivity)
        self.log_diag_covariances = torch.nn.Parameter(torch.randn(num_components, input_dim))

    def forward(self, x):
        # Get the mixture weights from the gating network
        mixture_weights = self.gating_network(x)

        diagonal_covariances = torch.exp(self.log_diag_covariances)

        # Compute log probabilities for each Gaussian component with diagonal covariance
        gaussians = dist.Independent(dist.Normal(self.means, diagonal_covariances), 1)
        log_probs = gaussians.log_prob(x.unsqueeze(1))  # Shape: (batch_size, num_components)
        # Compute the weighted log likelihood
        weighted_log_probs = log_probs + torch.log(mixture_weights)
        return torch.logsumexp(weighted_log_probs, dim=1)

    def score_samples(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x, device=self.covariances.device)
        return self.forward(x).detach().cpu().numpy()

def train_mixture_of_experts(model, X_train, num_epochs=100, learning_rate=1e-3):
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Use negative log-likelihood as the loss function
    def loss_function(log_likelihood):
        # We want to maximize the log likelihood, hence minimize the negative log likelihood
        return -log_likelihood.mean()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass: compute the log likelihood for each sample in the training set
        log_likelihood = model(X_train)

        # Compute the loss (negative log likelihood)
        loss = loss_function(log_likelihood)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"\t\tEpoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


################################################################################
## MODEL TRAINING
################################################################################

def train_global_approx_cem(
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

    print("At start model.concept_embeddings.requires_grad =", model.concept_embeddings.requires_grad)
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
            ## Step 1: Train the underlying model without any approximations yet
            ####################################################################

            start_time = time.time()
            dynamic_epochs = config.get('dynamic_epochs', 0)
            if dynamic_epochs:
                print(
                    f"\tTraining up the global concept generator for {dynamic_epochs} epochs"
                )

                # Else it is time to train it
                if (not rerun) and "dynamic_model_path" in config and (
                    os.path.exists(os.path.join(
                        result_dir,
                        f'{config["dynamic_model_path"]}.pt'
                    ))
                ):
                    dynamic_model_saved_path = os.path.join(
                        result_dir or ".",
                        f'{config["dynamic_model_path"]}.pt'
                    )
                    # Then we simply load the model and proceed
                    print("\tFound cached concept model... loading it")
                    state_dict = torch.load(dynamic_model_saved_path)
                    model.load_state_dict(state_dict, strict=False)
                else:
                    # Set the mode to dynamic!
                    model.mode = 'joint_same' if config.get('joint_trained', False) else 'dynamic'
                    dynamic_callbacks, dynamic_ckpt_call = _make_callbacks(
                        config,
                        result_dir,
                        full_run_name,
                    )
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
                    _restore_checkpoint(
                        model=model,
                        max_epochs=dynamic_epochs,
                        ckpt_call=dynamic_ckpt_call,
                        trainer=dynamic_trainer,
                    )
                    # Restart the optimizer state!
                    opts = model.optimizers()
                    opts.load_state_dict(opt_configs['optimizer'].state_dict())
                    if 'lr_scheduler' in opt_configs:
                        lr_scheduler = model.lr_schedulers()
                        lr_scheduler.load_state_dict(opt_configs['lr_scheduler'].state_dict())
                        lr_scheduler._reset()

                    training_time += time.time() - start_time
                    num_epochs += dynamic_trainer.current_epoch
                    start_time = time.time()
                    print(
                        f"\t\tDone after {num_epochs} epochs and {training_time} secs"
                    )

                    if "dynamic_model_path" in config:
                        dynamic_model_saved_path = os.path.join(
                            result_dir or ".",
                            f'{config["dynamic_model_path"]}.pt'
                        )
                        torch.save(
                            model.state_dict(),
                            dynamic_model_saved_path,
                        )

            eval_trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                logger=False,
            )
            # print("Trainable parameters are:")
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print("\t", name)
            model.mode = 'joint_same' if config.get('joint_trained', False) else 'dynamic'
            print("Validation eval after dynamic training:", evaluate.evaluate_cbm(
                model=model,
                trainer=eval_trainer,
                config=config,
                run_name=run_name,
                old_results=old_results,
                rerun=rerun,
                test_dl=val_dl,
                dl_name="val",
            ))

            ####################################################################
            ## Step 2: Learn prototypes
            ####################################################################

            # # Then produce the embeddings for the entire training set
            # prediction_trainer = pl.Trainer(
            #     accelerator=accelerator,
            #     devices=devices,
            #     logger=False,
            # )
            # model.mode = 'ground_truth'
            # model.output_embeddings = True
            # print(f"\t\tGenerating embeddings...")
            fast_train_loader = torch.utils.data.DataLoader(
                train_dl.dataset,
                batch_size=data_utils._largest_divisor(len(train_dl.dataset), max_val=512),
                num_workers=4,
                shuffle=False,
            )
            # train_batch_embs = prediction_trainer.predict(
            #     model,
            #     fast_train_loader,
            # )
            # pos_embeddings = torch.concat(
            #     [x[-2] for x in train_batch_embs],
            #     dim=0,
            # )
            # neg_embeddings = torch.concat(
            #     [x[-1] for x in train_batch_embs],
            #     dim=0,
            # )
            # train_bottlenecks = torch.concat(
            #     [x[1] for x in train_batch_embs],
            #     dim=0,
            # )
            # train_embs = torch.concat(
            #     [pos_embeddings, neg_embeddings],
            #     dim=-1,
            # )

            # train_probs = torch.concat(
            #     [x[0] for x in train_batch_embs],
            #     dim=0,
            # ).detach().cpu().numpy()

            # # # print("Trainable parameters are:")
            # # # for name, param in model.named_parameters():
            # # #     if param.requires_grad:
            # # #         print("\t", name)
            # # centroids = train_embs.mean(0).unsqueeze(0)
            # # dists = torch.norm(train_embs - centroids, dim=-1, p=2)
            # # thresholds = []
            # # for concept_idx in range(n_concepts):
            # #     concept_dists = dists[:, concept_idx].detach().cpu()
            # #     print("For concept", concept_idx, "mean distance is", torch.mean(concept_dists), ", max distance:", torch.max(concept_dists), "and min distance", torch.min(concept_dists))
            # #     init_threshdold = np.percentile(
            # #         concept_dists,
            # #         config.get('init_threshold_percentile', 90),
            # #     )
            # #     print("\tinit_threshdold is", init_threshdold)
            # #     thresholds.append(np.log(init_threshdold))
            # # print("thresholds =", thresholds)
            # # with torch.no_grad():
            # #     model.ood_centroids[:] = train_embs.mean(0)
            # #     model.ood_thresholds[:] = torch.FloatTensor(thresholds)
            # # if config.get('freeze_centroids', False):
            # #     model.ood_centroids.requires_grad = False
            # # model.mode = 'joint_same' if config.get('joint_trained', False) else 'dynamic'
            # # model.output_embeddings = False

            # # # Fit GMM to the training embeddings
            # # # gmms = []
            # # pos_gmms = []
            # # neg_gmms = []
            # # mixed_gmms = []
            # # pos_thresholds = []
            # # neg_thresholds = []
            # # mixed_thresholds = []
            # # _, _, c_train = data_utils.daloader_to_memory(
            # #     train_dl,
            # #     as_torch=False,
            # # )
            # # train_bottlenecks = train_bottlenecks.view(train_bottlenecks.shape[0], model.n_concepts, model.emb_size)
            # # for concept_idx in range(model.n_concepts):
            # #     if concept_idx == 0: print("pos_embeddings.shape =", pos_embeddings.shape)
            # #     pos_examples = pos_embeddings[c_train[:, concept_idx] == 1, concept_idx, :]
            # #     if concept_idx == 0: print("pos_examples.shape =", pos_examples.shape)
            # #     if concept_idx == 0: print("neg_embeddings.shape =", neg_embeddings.shape)
            # #     neg_examples = neg_embeddings[c_train[:, concept_idx] == 0, concept_idx, :]
            # #     if concept_idx == 0: print("neg_examples.shape =", neg_examples.shape)
            # #     gmm_pos = GaussianMixture(n_components=config.get('mixture_components', config['n_concept_variants']), covariance_type='full', reg_covar=1e-5, init_params='kmeans', random_state=42)
            # #     gmm_pos.fit(pos_examples)
            # #     gmm_neg = GaussianMixture(n_components=config.get('mixture_components', config['n_concept_variants']), covariance_type='full', reg_covar=1e-5, init_params='kmeans', random_state=42)
            # #     gmm_neg.fit(neg_examples)
            # #     pos_gmms.append(gmm_pos)
            # #     neg_gmms.append(gmm_neg)
            # #     if concept_idx == 0: print("Sorted positive scores:", sorted(list(gmm_pos.score_samples(pos_examples))))
            # #     pos_thresholds.append(np.percentile(list(gmm_pos.score_samples(pos_examples)), config.get('init_threshold_percentile', 5)))
            # #     if concept_idx == 0: print("\tpos_thresholds[-1]:", pos_thresholds[-1])
            # #     neg_thresholds.append(np.percentile(list(gmm_neg.score_samples(neg_examples)), config.get('init_threshold_percentile', 5)))

            # #     mixed_gmm = GaussianMixture(n_components=config.get('global_mixture_components', config['n_concept_variants']*2), reg_covar=1e-5, covariance_type='full', random_state=42)
            # #     mixed_gmm.fit(train_bottlenecks[:, concept_idx, :])
            # #     mixed_gmms.append(mixed_gmm)
            # #     mixed_thresholds.append(np.percentile(mixed_gmm.score_samples(train_bottlenecks[:, concept_idx, :]), config.get('init_threshold_percentile', 5)))

            # # thresholds = pos_thresholds + neg_thresholds + mixed_thresholds
            # # # model.set_gmm(gmms)
            # # model.set_gmm(pos_gmms=pos_gmms, neg_gmms=neg_gmms, mixed_gmms=mixed_gmms)
            # # model.gmm_thresholds = thresholds
            # # gmm_model_path = os.path.join(
            # #     result_dir or ".",
            # #     f'{full_run_name}_GMM_model.pt'
            # # )
            # # joblib.dump(pos_gmms + neg_gmms + mixed_gmms, gmm_model_path)

            # # gmm_threshold_path = os.path.join(
            # #     result_dir or ".",
            # #     f'{full_run_name}_GMM_threshold.pt'
            # # )
            # # joblib.dump(model.gmm_thresholds, gmm_threshold_path)

            # # Let's fit the beta distribution
            # if config.get('mode', 'ood_same') == 'ood_beta':


            #     alphas = []
            #     betas = []
            #     beta_threshs = []
            #     prob_gmms = []
            #     prob_threshs = []
            #     for concept_idx in range(train_probs.shape[-1]):
            #         concept_probs = train_probs[:, concept_idx:concept_idx+1]
            #         concept_probs = np.log(concept_probs / (1 - concept_probs + 1e-8))
            #         print("concept_idx =", concept_idx)
            #         print("\tconcept_probs =", concept_probs)


            #         # prob_gmm = MixtureOfExperts(concept_probs.shape[1], 2)
            #         # train_mixture_of_experts(prob_gmm, torch.FloatTensor(concept_probs), num_epochs=500, learning_rate=1e-3)
            #         # prob_gmms.append(prob_gmm)

            #         prob_gmm = GaussianMixture(n_components=2, reg_covar=1e-6, covariance_type='diag', random_state=42)
            #         prob_gmm.fit(concept_probs)
            #         prob_gmms.append(prob_gmm)
            #         scores = prob_gmm.score_samples(concept_probs)
            #         print("\tscores =", scores)
            #         # print("prob_gmm.covariances_ =", prob_gmm.covariances_)
            #         # print("prob_gmm.means_ =", prob_gmm.means_)
            #         # print("prob_gmm.weights_ =", prob_gmm.weights_)
            #         prob_threshs.append(np.percentile(scores, config.get('init_threshold_percentile', 5)))
            #         print("\tthresh =", prob_threshs[-1])

            #         # print("for concept", concept_idx, "and train_probs.shape =", train_probs.shape)
            #         # concept_probs = np.clip(train_probs[:, concept_idx], 1e-5, 1 - 1e-5)
            #         # print("\tmin(concept_probs) =", np.min(concept_probs))
            #         # print("\tmax(concept_probs) =", np.max(concept_probs))
            #         # alpha, beta, loc, scale = beta_fn.fit(concept_probs, floc=0, fscale=1)
            #         # print("\talpha =", alpha)
            #         # print("\tbeta =", beta)
            #         # print("\tloc =", loc)
            #         # print("\tscale =", scale)
            #         # alphas.append(alpha)
            #         # betas.append(beta)
            #         # log_likelihoods = beta_fn.logpdf(concept_probs, alpha, beta, loc=loc, scale=scale)
            #         # print("concept_probs =", concept_probs)
            #         # print("log_likelihoods =", log_likelihoods)
            #         # beta_threshs.append(np.percentile(log_likelihoods, config.get('init_threshold_percentile', 5)))
            #         # print("\tthresh =", beta_threshs[-1])
            #     model.prob_gmms = prob_gmms
            #     model.prob_threshs = prob_threshs
            #     prob_gmm_model_path = os.path.join(
            #         result_dir or ".",
            #         f'{full_run_name}_Prob_GMM_model.pt'
            #     )
            #     joblib.dump(prob_gmms, prob_gmm_model_path)

            #     prob_gmm_threshold_path = os.path.join(
            #         result_dir or ".",
            #         f'{full_run_name}_Prob_GMM_threshold.pt'
            #     )
            #     joblib.dump(prob_threshs, prob_gmm_threshold_path)
            #     # alphas = np.array(alphas)
            #     # betas = np.array(betas)
            #     # beta_threshs = np.array(beta_threshs)
            #     # print("beta_threshs =", beta_threshs)
            #     # with torch.no_grad():
            #     #     model.concept_log_alphas[:] = torch.log(torch.FloatTensor(alphas))
            #     #     model.concept_log_betas[:] = torch.log(torch.FloatTensor(betas))
            #     #     model.beta_thresholds[:] = torch.FloatTensor(beta_threshs)

            gmm_epochs = config.get('gmm_epochs')
            if gmm_epochs and (config.get('mode', 'ood_same') == 'ood_beta'):
                print(
                    f"\tTraining GMM models for {gmm_epochs} epochs"
                )
                # Set the mode to dynamic!
                model.mode = 'ood_beta'
                model.freeze_non_ood_weights(freeze_global_concept_generators=False)

                gmm_callbacks, gmm_ckpt_call = _make_callbacks(
                    config,
                    result_dir,
                    full_run_name,
                )
                print(
                    "[Number of parameters in GMM models",
                    sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "]"
                )
                print(
                    "[Number of non-trainable parameters in GMM models",
                    sum(p.numel() for p in model.parameters() if not p.requires_grad),
                    "]",
                )
                print("Trainable parameters are:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print("\t", name)
                opt_configs = model.configure_optimizers()
                gmm_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=gmm_epochs,
                    check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                    callbacks=gmm_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )
                gmm_trainer.fit(model, train_dl, val_dl)
                _check_interruption(gmm_trainer)
                training_time += time.time() - start_time
                num_epochs += gmm_trainer.current_epoch
                _restore_checkpoint(
                    model=model,
                    max_epochs=gmm_epochs,
                    ckpt_call=gmm_ckpt_call,
                    trainer=gmm_trainer,
                )
                model.trained_gmms = True
                model.unfreeze_non_ood_weights()
                # model.beta_thresholds.requires_grad = False
                for submodel in model.prob_gmms:
                    for param in submodel.parameters():
                        param.requires_grad = False

            ####################################################################
            ## Step 3: Train the approximation model
            ####################################################################

            start_time = time.time()
            approx_epochs = config.get('approx_epochs')
            if approx_epochs:
                print(
                    f"\tTraining approximation model for {approx_epochs} epochs"
                )
                # Set the mode to dynamic!
                model.mode = 'approx'
                if config.get('freeze_underlying_model', True):
                    model.freeze_non_approx_weights()

                approx_callbacks, approx_ckpt_call = _make_callbacks(
                    config,
                    result_dir,
                    full_run_name,
                )
                print(
                    "[Number of parameters in approximation model",
                    sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "]"
                )
                print(
                    "[Number of non-trainable parameters in approximation model",
                    sum(p.numel() for p in model.parameters() if not p.requires_grad),
                    "]",
                )
                # print("Trainable parameters are:")
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print("\t", name)
                print("model.concept_embeddings.requires_grad =", model.concept_embeddings.requires_grad)
                opt_configs = model.configure_optimizers()
                approx_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=approx_epochs,
                    check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                    callbacks=approx_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )

                approx_trainer.fit(model, train_dl, val_dl)
                _check_interruption(approx_trainer)
                training_time += time.time() - start_time
                num_epochs += approx_trainer.current_epoch
                _restore_checkpoint(
                    model=model,
                    max_epochs=approx_epochs,
                    ckpt_call=approx_ckpt_call,
                    trainer=approx_trainer,
                )
                if config.get('freeze_underlying_model', True):
                    model.unfreeze_non_approx_weights()


            ####################################################################
            ## Step 4: Finetune the entire model jointly
            ####################################################################

            start_time = time.time()
            finetune_epochs = config.get('finetuning_epochs', 0)
            if finetune_epochs:
                print(
                    f"\tFine-tuning end-to-end model for {finetune_epochs} epochs"
                )
                # Set the mode to dynamic!
                model.mode = 'approx'
                prev_l2_dist_loss_weight = model.l2_dist_loss_weight
                model.l2_dist_loss_weight = 0

                finetune_callbacks, finetune_ckpt_call = _make_callbacks(
                    config,
                    result_dir,
                    full_run_name,
                )
                opt_configs = model.configure_optimizers()
                finetune_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=finetune_epochs,
                    check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                    callbacks=finetune_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )
                print(
                    "[Number of parameters in approximation model",
                    sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "]"
                )
                print(
                    "[Number of non-trainable parameters in approximation model",
                    sum(p.numel() for p in model.parameters() if not p.requires_grad),
                    "]",
                )
                # print("Trainable parameters are:")
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print("\t", name)
                finetune_trainer.fit(model, train_dl, val_dl)
                _check_interruption(finetune_trainer)
                training_time += time.time() - start_time
                num_epochs += finetune_trainer.current_epoch
                _restore_checkpoint(
                    model=model,
                    max_epochs=finetune_epochs,
                    ckpt_call=finetune_ckpt_call,
                    trainer=finetune_trainer,
                )
                model.l2_dist_loss_weight = prev_l2_dist_loss_weight

            ####################################################################
            ## Step 5: Train OOD ball detector for fallback
            ####################################################################

            start_time = time.time()
            ood_epochs = config.get('ood_epochs', 0)
            if ood_epochs:
                print(
                    f"\tTraining ODD detection model {ood_epochs} epochs"
                )
                # Set the mode to dynamic!
                model.mode = config.get('mode', 'ood_same')
                model.ood_centroids.requires_grad = False
                model.ood_thresholds.requires_grad = False
                model.ood_fn_scales.requires_grad = False
                if config.get('ood_freeze_underlying_model', True):
                    model.freeze_non_ood_weights(
                        freeze_label_predictor=config.get('ood_freeze_label_predictor', True),
                        freeze_concept_rank_model=config.get('ood_freeze_concept_rank_model', True),
                        freeze_global_concept_generators=config.get('ood_freeze_global_concept_generators', True),
                    )
                print("Train eval after before OOD training:", evaluate.evaluate_cbm(
                    model=model,
                    trainer=eval_trainer,
                    config=config,
                    run_name=run_name,
                    old_results=old_results,
                    rerun=rerun,
                    test_dl=fast_train_loader,
                    dl_name="train",
                ))

                ood_callbacks, ood_ckpt_call = _make_callbacks(
                    config,
                    result_dir,
                    full_run_name,
                )
                opt_configs = model.configure_optimizers()
                ood_trainer = pl.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    max_epochs=ood_epochs,
                    check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                    callbacks=ood_callbacks,
                    logger=logger or False,
                    enable_checkpointing=enable_checkpointing,
                    gradient_clip_val=gradient_clip_val,
                )
                print(
                    "[Number of parameters in OOD model",
                    sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "]"
                )
                print(
                    "[Number of non-trainable parameters in OOD model",
                    sum(p.numel() for p in model.parameters() if not p.requires_grad),
                    "]",
                )
                print("Trainable parameters are:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print("\t", name)
                ood_trainer.fit(model, train_dl, val_dl)
                _check_interruption(ood_trainer)
                training_time += time.time() - start_time
                num_epochs += ood_trainer.current_epoch
                _restore_checkpoint(
                    model=model,
                    max_epochs=ood_epochs,
                    ckpt_call=ood_ckpt_call,
                    trainer=ood_trainer,
                )
                if config.get('ood_freeze_underlying_model', True):
                    model.unfreeze_non_ood_weights()

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