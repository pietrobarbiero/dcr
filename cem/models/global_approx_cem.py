import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch

from scipy.stats import beta as beta_fn

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


from torchvision.models import resnet50

from cem.models.intcbm import IntAwareConceptEmbeddingModel
import cem.train.utils as utils


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
        # print("diagonal_covariances =", diagonal_covariances)
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

class GlobalApproxConceptEmbeddingModel(IntAwareConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        concept_loss_weight=1,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,

        top_k_accuracy=None,

        intervention_task_discount=1.1,
        intervention_weight=5,
        concept_map=None,
        use_concept_groups=True,

        rollout_init_steps=0,
        int_model_layers=None,
        int_model_use_bn=True,
        num_rollouts=1,

        # Parameters regarding how we select how many concepts to intervene on
        # in the horizon of a current trajectory (this is the lenght of the
        # trajectory)
        max_horizon=6,
        initial_horizon=2,
        horizon_rate=1.005,

        # Experimental/debugging arguments
        intervention_discount=1,
        include_only_last_trajectory_loss=True,
        task_loss_weight=1,
        intervention_task_loss_weight=1,

        ##################################
        # New arguments
        #################################
        temperature=1,
        n_concept_variants=5,
        initial_concept_embeddings=None,
        fixed_embeddings=False,
        mode='approx',
        l2_dist_loss_weight=0,
        compression_mode='learnt',
        attention_fn='softmax',
        ood_centroids=None,
        learnable_centroids=True,
        ood_thresholds=None,
        learnable_thresholds=True,
        log_thresholds=True,
        thresh_l2_loss=0,  # L2 for the threshold values so that they don't just all become massive and instead attempt to fairly capture the ball containing the distribution
        ood_dropout_prob=0,  # Probability we will randomly use the global embedding when training in OOD mode regarding the distance and the threshold
        use_dynamic_for_probs=False,
        distance_l2_loss=0, # L2 penalty to move the centroid prototypes closer to the distribution

        gmms=None,
        gmm_thresholds=None,
        global_mixture_components=None,
        prob_gmms=None,
        prob_threshs=None,
        ood_loss_weight=0,

        approx_prediction_mode='same',
    ):
        super(GlobalApproxConceptEmbeddingModel, self).__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            concept_loss_weight=concept_loss_weight,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            c_extractor_arch=c_extractor_arch,
            output_latent=output_latent,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            top_k_accuracy=top_k_accuracy,
            intervention_task_discount=intervention_task_discount,
            intervention_weight=intervention_weight,
            concept_map=concept_map,
            use_concept_groups=use_concept_groups,
            rollout_init_steps=rollout_init_steps,
            int_model_layers=int_model_layers,
            int_model_use_bn=int_model_use_bn,
            num_rollouts=num_rollouts,
            max_horizon=max_horizon,
            initial_horizon=initial_horizon,
            horizon_rate=horizon_rate,
            intervention_discount=intervention_discount,
            include_only_last_trajectory_loss=include_only_last_trajectory_loss,
            task_loss_weight=task_loss_weight,
            intervention_task_loss_weight=intervention_task_loss_weight,
        )
        assert mode in ['dynamic', 'approx', 'ood', 'ood_fixed', 'prototypes' ,'joint', 'joint_same', 'ood_same', 'ood_beta', 'separator']
        self.mode = mode
        self.compression_mode = compression_mode
        self.temperature = temperature
        self.n_concept_variants = n_concept_variants
        self.l2_dist_loss_weight = l2_dist_loss_weight
        self.attention_fn = attention_fn
        self.use_dynamic_for_probs = use_dynamic_for_probs
        self.distance_l2_loss = distance_l2_loss
        # OOD Detection
        if ood_centroids is None:
            ood_centroids = torch.normal(
                torch.zeros(self.n_concepts, 2 * emb_size),
                torch.ones(self.n_concepts, 2 * emb_size),
            )
        self.ood_centroids = torch.nn.Parameter(
            ood_centroids,
            requires_grad=learnable_centroids,
        )

        if ood_thresholds is None:
            ood_thresholds = 5*torch.rand(
                self.n_concepts,
            )
        self.log_thresholds = log_thresholds
        self.thresh_l2_loss = thresh_l2_loss
        self.ood_thresholds = torch.nn.Parameter(
            ood_thresholds,
            requires_grad=learnable_thresholds,
        )

        self.ood_fn_scales = torch.nn.Parameter(
             torch.rand(
                self.n_concepts,
            ),
            requires_grad=True,
        )
        self.ood_dropout_prob = ood_dropout_prob
        self.ood_dropout = torch.nn.Dropout(
            p=(1 - ood_dropout_prob),  # We do 1 - ood_prob as this will be applied to the selection of the global embedding
        )

        # Must output logits with shape (B, n_concept_variants)
        if (initial_concept_embeddings is False) or (
            initial_concept_embeddings is None
        ):
            initial_concept_embeddings = torch.normal(
                torch.zeros(self.n_concepts, n_concept_variants, 2, emb_size),
                torch.ones(self.n_concepts, n_concept_variants, 2, emb_size),
            )
        else:
            if isinstance(initial_concept_embeddings, np.ndarray):
                initial_concept_embeddings = torch.FloatTensor(
                    initial_concept_embeddings
                )
            emb_size = initial_concept_embeddings.shape[-1]
        self.concept_embeddings = torch.nn.Parameter(
            initial_concept_embeddings,
            requires_grad=(not fixed_embeddings),
        )
        # if n_concept_variants == 1:
        #     # Then no real need for a fancy model as the selection is
        #     # trivial
        #     self.attn_model = [
        #         lambda x: torch.ones((x.shape[0], 1), device=x.device)
        #         for _ in range(self.n_concepts)
        #     ]
        # else:
        self.attn_model = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(
                    list(
                        self.pre_concept_model.modules()
                    )[-1].out_features,
                    64,
                ),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    64,
                    32,
                ),
                torch.nn.Linear(
                    32,
                    self.n_concept_variants,
                ),
            )
            for _ in range(self.n_concepts)
        ])

        self._dynamic_contexts = None
        self._global_contexts = None
        self._distances = None
        self._global_selected = None

        self.gmm_weights = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, 2*n_concept_variants),
                torch.ones(n_concepts, 2*n_concept_variants),
            ),
            requires_grad=False,
        )
        self.gmm_means = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, 2*n_concept_variants, emb_size),
                torch.ones(n_concepts, 2*n_concept_variants, emb_size),
            ),
            requires_grad=False,
        )
        self.gmm_covariances = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, 2*n_concept_variants, emb_size, emb_size),
                torch.ones(n_concepts, 2*n_concept_variants, emb_size, emb_size),
            ),
            requires_grad=False,
        )

        global_mixture_components = global_mixture_components or 2*n_concept_variants
        self.mixed_gmm_weights = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, global_mixture_components),
                torch.ones(n_concepts, global_mixture_components),
            ),
            requires_grad=False,
        )
        self.mixed_gmm_means = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, global_mixture_components, emb_size),
                torch.ones(n_concepts, global_mixture_components, emb_size),
            ),
            requires_grad=False,
        )
        self.mixed_gmm_covariances = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, global_mixture_components, emb_size, emb_size),
                torch.ones(n_concepts, global_mixture_components, emb_size, emb_size),
            ),
            requires_grad=False,
        )

        self.gmms = None
        self.mixed_gmms = None
        self.gmm_thresholds = None

        if gmms is not None:
            # self.set_gmm(gmms)

            self.set_gmm(
                pos_gmms=gmms[:self.n_concepts],
                neg_gmms=gmms[self.n_concepts:2*self.n_concepts],
                mixed_gmms=gmms[2*self.n_concepts:],
            )

        if gmm_thresholds is not None:
            self.gmm_thresholds = gmm_thresholds

        # c2y model for global embedding
        self.approx_prediction_mode = approx_prediction_mode
        if approx_prediction_mode == 'same':
            self.c2y_model_approx = self.c2y_model
        elif approx_prediction_mode == 'new':
            units = [
                n_concepts * emb_size
            ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model_approx = torch.nn.Sequential(*layers)


            # And the probability generators
            self.global_concept_prob_generators = torch.nn.ModuleList()
            for i in range(n_concepts):
                if self.shared_prob_gen and (
                    len(self.global_concept_prob_generators) == 0
                ):
                    # Then we will use one and only one probability generator which
                    # will be shared among all concepts. This will force concept
                    # embedding vectors to be pushed into the same latent space
                    self.global_concept_prob_generators.append(torch.nn.Linear(
                        2 * emb_size,
                        1,
                    ))
                elif not self.shared_prob_gen:
                    self.global_concept_prob_generators.append(torch.nn.Linear(
                        2 * emb_size,
                        1,
                    ))
        else:
            raise ValueError(
                f'Only approx_prediction_mode modes allowed are "same" '
                f'and "new".'
            )


        # self.concept_log_alphas = torch.nn.Parameter(
        #     torch.normal(
        #         torch.zeros(n_concepts),
        #         torch.ones(n_concepts),
        #     ),
        #     requires_grad=False,  # TODO: we may be able to learn these end-to-end!
        # )

        # self.concept_log_betas = torch.nn.Parameter(
        #     torch.normal(
        #         torch.zeros(n_concepts),
        #         torch.ones(n_concepts),
        #     ),
        #     requires_grad=False,  # TODO: we may be able to learn these end-to-end!
        # )
        self.beta_thresholds = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts),
                torch.ones(n_concepts),
            ),
            requires_grad=True,  # TODO: we may be able to learn these end-to-end!
        )
        # self.prob_gmms = prob_gmms
        # self.prob_threshs = prob_threshs

        self.prob_gmms = torch.nn.ModuleList([
            MixtureOfExperts(1, 2)
            for _ in range(n_concepts)
        ])
        self._combined_log_likelihoods = None
        self.ood_loss_weight = ood_loss_weight

        # self.concept_locs = torch.nn.Parameter(
        #     torch.normal(
        #         torch.zeros(n_concepts),
        #         torch.ones(n_concepts),
        #     ),
        #     requires_grad=False,  # TODO: we may be able to learn these end-to-end!
        # )
        # self.concept_scales = torch.nn.Parameter(
        #     torch.normal(
        #         torch.zeros(n_concepts),
        #         torch.ones(n_concepts),
        #     ),
        #     requires_grad=False,  # TODO: we may be able to learn these end-to-end!
        # )

        self.separators = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, emb_size),
                torch.ones(n_concepts, emb_size),
            ),
            requires_grad=True,
        )

    def _construct_c2y_input(
        self,
        pos_embeddings,
        neg_embeddings,
        probs,
        **task_loss_kwargs,
    ):
        bottleneck = (
            pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                neg_embeddings * (
                    1 - torch.unsqueeze(probs, dim=-1)
                )
            )
        )
        if (self.mode == 'joint') or (
            (self.mode == 'ood') and (self.approx_prediction_mode == 'new')
        ) or (
            (self.mode == 'ood_fixed') and (self.approx_prediction_mode == 'new')
        ):
                bottleneck = bottleneck.view(
                (-1, 2 * self.emb_size * self.n_concepts)
            )
        else:
            bottleneck = bottleneck.view(
                (-1, self.emb_size * self.n_concepts)
            )
        return bottleneck

    def _construct_rank_model_input(self, bottleneck, prev_interventions):
        if (self.mode == 'joint') or (
            (self.mode == 'ood') and (self.approx_prediction_mode == 'new')
        ) or (
             (self.mode == 'ood_fixed') and (self.approx_prediction_mode == 'new')
        ):
            bottleneck = bottleneck.reshape(bottleneck.shape[0], self.n_concepts, 2*self.emb_size)
            bottleneck = bottleneck[:, :, self.emb_size:].reshape(bottleneck.shape[0], -1)
        cat_inputs = [
            bottleneck.view(bottleneck.shape[0], -1),
            prev_interventions,
        ]
        return torch.concat(
            cat_inputs,
            dim=-1,
        )

    def _predict_labels(self, bottleneck, **task_loss_kwargs):
        if self.mode in ['dynamic', 'ground_truth', 'joint_same', 'ood_same', 'ood_beta']:
            return self.c2y_model(bottleneck)
        if self.mode == 'approx':
            return self.c2y_model_approx(bottleneck)
        if self.mode == 'joint':
            total_embs = bottleneck.view(bottleneck.shape[0], self.n_concepts, 2*self.emb_size)
            dynamic_bottleneck = total_embs[:, :, self.emb_size:].reshape(bottleneck.shape[0], -1)
            dynamic_outputs = self.c2y_model(dynamic_bottleneck)

            global_bottleneck = total_embs[:, :, :self.emb_size].reshape(bottleneck.shape[0], -1)
            global_outputs = self.c2y_model_approx(global_bottleneck)

            out_value = (
                0.5 * global_outputs +
                0.5 * dynamic_outputs
            )
            # print("out_value.shape =", out_value.shape)
            return out_value

        if self.mode in ['ood', 'ood_fixed']:
            if self.approx_prediction_mode == 'new':
                total_embs = bottleneck.view(bottleneck.shape[0], self.n_concepts, 2*self.emb_size)
                dynamic_bottleneck = total_embs[:, :, self.emb_size:].reshape(bottleneck.shape[0], -1)
                dynamic_outputs = self.c2y_model(dynamic_bottleneck)

                global_bottleneck = total_embs[:, :, :self.emb_size].reshape(bottleneck.shape[0], -1)
                global_outputs = self.c2y_model_approx(global_bottleneck)

                # TODO: make the global selection be on a single model of the likelihood of the dynamic bottleneck alone
                global_selected = torch.mean(self._global_selected, dim=1)
                # print("Selected", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples with a max value of", torch.max(global_selected).detach().cpu().numpy(), "an a min value of", torch.min(global_selected).detach().cpu().numpy())
                out_value = (
                    global_selected.unsqueeze(-1) * global_outputs +
                    (1 - global_selected.unsqueeze(-1)) * dynamic_outputs
                )
                out_value = (
                    0.5 * global_outputs +
                    0.5 * dynamic_outputs
                )
                # print("out_value.shape =", out_value.shape)
                return out_value
            else:
                return self.c2y_model(bottleneck)

        raise ValueError(
            f'Unsupported mode "{self.mode}"'
        )


    def set_gmm(self, pos_gmms, neg_gmms, mixed_gmms=None):
        self.gmms = pos_gmms + neg_gmms
        self.mixed_gmms = mixed_gmms
        with torch.no_grad():
            for i, (pos_gmm, neg_gmm) in enumerate(zip(pos_gmms, neg_gmms)):
                self.gmm_weights[i, :self.n_concept_variants] = torch.tensor(pos_gmm.weights_[:self.n_concept_variants] , dtype=torch.float32)
                self.gmm_weights[i, self.n_concept_variants:] = torch.tensor(neg_gmm.weights_ [:self.n_concept_variants], dtype=torch.float32)
                self.gmm_means[i, :self.n_concept_variants, :] = torch.tensor(pos_gmm.means_[:self.n_concept_variants, :], dtype=torch.float32)
                self.gmm_means[i, self.n_concept_variants:, :] = torch.tensor(neg_gmm.means_[:self.n_concept_variants, :], dtype=torch.float32)
                self.gmm_covariances[i, :self.n_concept_variants, :, :] = torch.tensor(pos_gmm.covariances_[:self.n_concept_variants, :, :], dtype=torch.float32)
                self.gmm_covariances[i, self.n_concept_variants:, :, :] = torch.tensor(neg_gmm.covariances_[:self.n_concept_variants, :, :], dtype=torch.float32)
                self.concept_embeddings[i, :, 0, :] = self.gmm_means[i, :self.n_concept_variants, :]
                self.concept_embeddings[i, :, 1, :] = self.gmm_means[i, self.n_concept_variants:, :]

                if mixed_gmms is not None:
                    self.mixed_gmm_weights[i, :] = torch.tensor(self.mixed_gmms[i].weights_ , dtype=torch.float32)
                    self.mixed_gmm_means[i, :, :] = torch.tensor(self.mixed_gmms[i].means_, dtype=torch.float32)
                    self.mixed_gmm_covariances[i, :, :, :] = torch.tensor(self.mixed_gmms[i].covariances_, dtype=torch.float32)

    def _gmm_log_likelihood(self, x, concept_idx, mixed=False):
        if mixed:
            weights = self.mixed_gmm_weights[concept_idx, :]
            means = self.mixed_gmm_means[concept_idx, :, :]
            covariances = self.mixed_gmm_covariances[concept_idx, :, :, :]
        else:
            weights = self.gmm_weights[concept_idx, :]
            means = self.gmm_means[concept_idx, :, :]
            covariances = self.gmm_covariances[concept_idx, :, :, :]
        num_components = len(weights)
        num_samples = x.shape[0]
        log_likelihood = torch.zeros(num_samples).to(x.device)
        for k in range(num_components):
            # Define the multivariate normal for each component
            mvn = MultivariateNormal(means[k], covariances[k])

            # Compute the weighted log probability for each component
            log_prob_k = mvn.log_prob(x) + torch.log(weights[k])

            # Add the component log likelihood to the total
            log_likelihood += torch.exp(log_prob_k)

        # Take log of the sum across components to get final log likelihood
        return torch.log(log_likelihood)

    def log_beta_likelihood(self, probs, concept_idx):
        new_probs = np.clip(probs[:, concept_idx].detach().cpu().numpy(), 1e-5, 1 - 1e-5)
        return torch.FloatTensor(
            beta_fn.logpdf(new_probs, torch.exp(self.concept_log_betas[concept_idx]).detach().cpu().numpy(), torch.exp(self.concept_log_alphas[concept_idx]).detach().cpu().numpy(), loc=0, scale=1)
        ).to(probs.device)
        beta = torch.exp(self.concept_log_betas[concept_idx])
        alpha = torch.exp(self.concept_log_alphas[concept_idx])
        concept_prob = probs[:, concept_idx]
        beta_val = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        return (alpha - 1) * torch.log(concept_prob) + (beta - 1)*torch.log(1 - concept_prob) - beta_val

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=None,
        output_interventions=None,
    ):
        if self.mode == 'ground_truth':
            intervention_idxs = torch.ones_like(c)

        return super(GlobalApproxConceptEmbeddingModel, self)._forward(
            x=x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            latent=latent,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=output_embeddings,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )
    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        extra_outputs = {}
        # print("Mode is:", self.mode)
        if self.mode in ['dynamic', 'ground_truth']:
            return  super(GlobalApproxConceptEmbeddingModel, self)._generate_concept_embeddings(
                x=x,
                latent=latent,
                training=training,
            )
        if latent is None:
            pre_c = self.pre_concept_model(x)
            dynamic_contexts = []
            global_contexts = []

            # First predict all the concept probabilities
            for concept_idx, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[concept_idx]
                dynamic_context = context_gen(pre_c)
                dynamic_contexts.append(torch.unsqueeze(dynamic_context, dim=1))
                # distr has shape (B, n_concept_variants)
                if self.attention_fn == 'sigmoid':
                    distr = torch.sigmoid(
                        self.temperature * self.attn_model[concept_idx](pre_c)
                    )
                elif self.attention_fn == 'softmax':
                    distr = torch.softmax(
                        self.temperature * self.attn_model[concept_idx](pre_c),
                        dim=-1,
                    )
                else:
                    raise ValueError(
                        f'Unrecognized attention_fn "{self.attention_fn}".'
                    )
                #  global_context_pos has shape (1, n_concept_variants, emb_size)
                global_context_pos = self.concept_embeddings[concept_idx:concept_idx+1, :, 0, :]
                #  global_context_pos has shape (B, n_concept_variants, emb_size)
                global_context_pos = global_context_pos.expand(pre_c.shape[0], -1, -1)
                #  global_context_pos now has shape (B, emb_size)
                global_context_pos = (
                    distr.unsqueeze(-1) * global_context_pos
                ).sum(1)

                #  global_context_neg has shape (1, n_concept_variants, emb_size)
                global_context_neg = self.concept_embeddings[concept_idx:concept_idx+1, :, 1, :]
                #  global_context_neg has shape (B, n_concept_variants, emb_size)
                global_context_neg = global_context_neg.expand(pre_c.shape[0], -1, -1)
                #  global_context_neg now has shape (B, emb_size)
                global_context_neg = (
                    distr.unsqueeze(-1) * global_context_neg
                ).sum(1)

                global_context = torch.concat(
                    [global_context_pos, global_context_neg],
                    dim=-1,
                )
                global_contexts.append(torch.unsqueeze(global_context, dim=1))
            dynamic_contexts = torch.cat(dynamic_contexts, axis=1)
            global_contexts = torch.cat(global_contexts, axis=1)
            latent = dynamic_contexts, global_contexts
        else:
            dynamic_contexts, global_contexts = latent

         # Now we can compute all the probabilites!
        c_sem = []
        c_logits = []
        if self.mode == 'joint':
            for concept_idx in range(self.n_concepts):
                if self.shared_prob_gen:
                    prob_dyn_gen = self.concept_prob_generators[0]
                    prob_global_gen = self.global_concept_prob_generators[0]
                else:
                    prob_dyn_gen = self.concept_prob_generators[concept_idx]
                    prob_global_gen = self.global_concept_prob_generators[concept_idx]

                prob = (
                    self.sig(prob_dyn_gen(dynamic_contexts[:, concept_idx, :])) +
                    self.sig(prob_global_gen(global_contexts[:, concept_idx, :]))
                ) / 2
                c_sem.append(prob)
        else:
            if self.use_dynamic_for_probs:
                prob_contexts = dynamic_contexts
            else:
                prob_contexts = global_contexts
            for concept_idx in range(self.n_concepts):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[concept_idx]
                c_logits.append(prob_gen(prob_contexts[:, concept_idx, :]))
                prob = self.sig(c_logits[-1])
                c_sem.append(prob)
        c_sem = torch.cat(c_sem, axis=-1)
        c_logits = torch.cat(c_logits, axis=-1)


        if self.mode == 'prototypes':
            pos_embeddings = dynamic_contexts[:, :, :self.emb_size]
            neg_embeddings = dynamic_contexts[:, :, self.emb_size:]
            if training:
                total_distances = []
                for concept_idx in range(self.n_concepts):
                    # if concept_idx == 0: print("\tself.ood_centroids[concept_idx, :10]: ", self.ood_centroids[concept_idx, :10])
                    # if concept_idx == 0: print("\tpre_c[0, :10]: ", pre_c[0, :10])
                    distances = torch.norm(
                        dynamic_contexts[:, concept_idx, :] - self.ood_centroids[concept_idx:concept_idx+1, :].expand(dynamic_contexts.shape[0], -1),
                        p=2,
                        dim=-1,
                    ).unsqueeze(-1)
                    # if concept_idx == 0: print("\tdistances[0] =", distances[0])
                    total_distances.append(distances)
                total_distances = torch.concat(total_distances, dim=1)
                self._distances = total_distances
        elif self.mode == 'approx':
            # Then we will use entirely the global contexts as we are learning
            # the global approximation
            pos_embeddings = global_contexts[:, :, :self.emb_size]
            neg_embeddings = global_contexts[:, :, self.emb_size:]
            if training:
                self._dynamic_contexts = dynamic_contexts
                self._global_contexts = global_contexts
        elif self.mode == 'joint':
            pos_embeddings = torch.concat(
                [
                    dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                    global_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                ],
                dim=-1,
            )
            neg_embeddings = torch.concat(
                [
                    dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
                    global_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
                ],
                dim=-1,
            )
        elif self.mode in ['joint_same', 'ood_same']:
            # Then we will generate a mixture of the dynamic and global
            # embedding weighted by the OOD weight
            pos_embeddings = []
            neg_embeddings = []
            # print("training is:", training)
            # print("self.gmms is:", self.gmms)
            # print("self.ood_thresholds is:", self.ood_thresholds)
            for concept_idx in range(self.n_concepts):
                if training and (self.mode != 'ood_same'):
                    pos_global_selected = torch.bernoulli(
                        torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device) * self.ood_dropout_prob,
                    )
                    neg_global_selected = torch.bernoulli(
                        torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device) * self.ood_dropout_prob,
                    )
                elif (self.mixed_gmms is not None) and (self.gmm_thresholds is not None):
                    mixed = (
                        dynamic_contexts[:, concept_idx, :self.emb_size] * c_sem[:, concept_idx:concept_idx+1] +
                        (1 - c_sem[:, concept_idx:concept_idx+1])*dynamic_contexts[:, concept_idx, self.emb_size:]
                    )
                    global_scores = torch.FloatTensor(self.mixed_gmms[concept_idx].score_samples(mixed.detach().cpu().numpy())).to(dynamic_contexts)
                    # global_scores = self._gmm_log_likelihood(
                    #     mixed,
                    #     concept_idx=concept_idx,
                    #     mixed=True,
                    # )
                    global_selected = torch.where(
                        self.gmm_thresholds[2*self.n_concepts + concept_idx] > global_scores.unsqueeze(-1).unsqueeze(-1),
                        torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                        torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                    )

                    pos_scores = torch.FloatTensor(self.gmms[concept_idx].score_samples(dynamic_contexts[:, concept_idx, :self.emb_size].detach().cpu().numpy())).to(dynamic_contexts)
                    # pos_scores = self._gmm_log_likelihood(dynamic_contexts[:, concept_idx, :self.emb_size], concept_idx=concept_idx)
                    neg_scores = torch.FloatTensor(self.gmms[concept_idx + self.n_concepts].score_samples(dynamic_contexts[:, concept_idx, self.emb_size:].detach().cpu().numpy())).to(dynamic_contexts)
                    # neg_scores = self._gmm_log_likelihood(dynamic_contexts[:, concept_idx, self.emb_size:], concept_idx=concept_idx)

                    pos_global_selected = torch.where(
                        # self.gmm_thresholds[concept_idx] > scores.unsqueeze(-1).unsqueeze(-1),
                        self.gmm_thresholds[concept_idx] > pos_scores.unsqueeze(-1).unsqueeze(-1),
                        torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                        torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                    )
                    pos_global_selected = (c_sem[:, concept_idx] >= 0.5).unsqueeze(-1).unsqueeze(-1) * pos_global_selected
                    neg_global_selected = torch.where(
                        # self.gmm_thresholds[concept_idx] > scores.unsqueeze(-1).unsqueeze(-1),
                        self.gmm_thresholds[concept_idx + self.n_concepts] > neg_scores.unsqueeze(-1).unsqueeze(-1),
                        torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                        torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                    )
                    neg_global_selected = (c_sem[:, concept_idx] < 0.5).unsqueeze(-1).unsqueeze(-1) * neg_global_selected
                    if concept_idx == 0 and (not training) and (torch.sum(c_sem[:, concept_idx] >= 0.5).detach().cpu().numpy() > 0): print("\tSelected positive global embedding for", torch.sum(pos_global_selected > 0.5).detach().cpu().numpy(), "out of", pos_global_selected.shape[0], "samples with a max value of ", torch.max(pos_scores[c_sem[:, concept_idx] >= 0.5]), "and a min value of", torch.min(pos_scores[c_sem[:, concept_idx] >= 0.5]), "when self.gmm_thresholds[concept_idx] =", self.gmm_thresholds[concept_idx])
                    if concept_idx == 0 and (not training): print("\tSelected negative global embedding for", torch.sum(neg_global_selected > 0.5).detach().cpu().numpy(), "out of", neg_global_selected.shape[0], "samples with a max value of ", torch.max(neg_scores[c_sem[:, concept_idx] < 0.5]), "and a min value of", torch.min(neg_scores[c_sem[:, concept_idx] < 0.5]), "when self.gmm_thresholds[concept_idx + self.n_concepts] =", self.gmm_thresholds[concept_idx + self.n_concepts])
                    if concept_idx == 0 and (not training): print("\tFirst 10 probs are:", c_sem[:, concept_idx][:10])
                    # global_selected = torch.logical_or(neg_global_selected)
                    # global_selected = torch.where(
                    #     # self.gmm_thresholds[concept_idx] > scores.unsqueeze(-1).unsqueeze(-1),
                    #     torch.logical_or(self.gmm_thresholds[concept_idx] > pos_scores.unsqueeze(-1).unsqueeze(-1), self.gmm_thresholds[concept_idx + self.n_concepts] > neg_scores.unsqueeze(-1).unsqueeze(-1)),
                    #     torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                    #     torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                    # )
                    # pos_global_selected = global_selected
                    # neg_global_selected = global_selected
                    if concept_idx == 0 and not training: print("\tTotal global selected embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples with a max value of ", torch.max(global_scores), "and a min value of", torch.min(global_scores), "when self.gmm_thresholds[2*self.n_concepts + concept_idx] =", self.gmm_thresholds[2*self.n_concepts + concept_idx])


                elif (self.gmms is not None) and (self.gmm_thresholds is not None):
                    # pos_scores = torch.FloatTensor(self.gmms[concept_idx].score_samples(dynamic_contexts[:, concept_idx, :self.emb_size].detach().cpu().numpy())).to(dynamic_contexts)
                    pos_scores = self._gmm_log_likelihood(dynamic_contexts[:, concept_idx, :self.emb_size], concept_idx=concept_idx)
                    # neg_scores = torch.FloatTensor(self.gmms[concept_idx + self.n_concepts].score_samples(dynamic_contexts[:, concept_idx, self.emb_size:].detach().cpu().numpy())).to(dynamic_contexts)
                    neg_scores = self._gmm_log_likelihood(dynamic_contexts[:, concept_idx, self.emb_size:], concept_idx=concept_idx)

                    pos_global_selected = torch.where(
                        # self.gmm_thresholds[concept_idx] > scores.unsqueeze(-1).unsqueeze(-1),
                        self.gmm_thresholds[concept_idx] > pos_scores.unsqueeze(-1).unsqueeze(-1),
                        torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                        torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                    )
                    neg_global_selected = torch.where(
                        # self.gmm_thresholds[concept_idx] > scores.unsqueeze(-1).unsqueeze(-1),
                        self.gmm_thresholds[concept_idx + self.n_concepts] > neg_scores.unsqueeze(-1).unsqueeze(-1),
                        torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                        torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device),
                    )

                    # if concept_idx == 0: print("\tSelected positive global embedding for", torch.sum(pos_global_selected > 0.5).detach().cpu().numpy(), "out of", pos_global_selected.shape[0], "samples with a max value of ", torch.max(pos_scores), "and a min value of", torch.min(pos_scores), "when self.gmm_thresholds[concept_idx] =", self.gmm_thresholds[concept_idx])
                    # if concept_idx == 0: print("\tSelected negative global embedding for", torch.sum(neg_global_selected > 0.5).detach().cpu().numpy(), "out of", neg_global_selected.shape[0], "samples with a max value of ", torch.max(neg_scores), "and a min value of", torch.min(neg_scores), "when self.gmm_thresholds[concept_idx + self.n_concepts] =", self.gmm_thresholds[concept_idx + self.n_concepts])
                else:
                    pos_global_selected = torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device)
                    neg_global_selected = torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device)


                if self.approx_prediction_mode == 'new':
                    pos_embeddings.append(
                        pos_global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
                        (1 - pos_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size]
                    )
                    neg_embeddings.append(
                        neg_global_selected * global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
                        (1 - neg_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:]
                    )

                else:
                    # At this point, we are ready to mix the output embeddings
                    pos_embeddings.append(
                        pos_global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
                        (1 - pos_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size]
                    )
                    neg_embeddings.append(
                        neg_global_selected * global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
                        (1 - neg_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:]
                    )

            pos_embeddings = torch.concat(pos_embeddings, dim=1)
            neg_embeddings = torch.concat(neg_embeddings, dim=1)


        elif self.mode in ['joint_same', 'ood_beta']:
            # Then we will generate a mixture of the dynamic and global
            # embedding weighted by the OOD weight
            pos_embeddings = []
            neg_embeddings = []
            combined_log_likelihoods = []
            for concept_idx in range(self.n_concepts):
                if training and (self.mode == 'joint_same'):
                    pos_global_selected = torch.bernoulli(
                        torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device) * self.ood_dropout_prob,
                    )
                    neg_global_selected = torch.bernoulli(
                        torch.ones((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device) * self.ood_dropout_prob,
                    )
                elif self.mode == 'ood_beta' and (self.prob_gmms is not None):
                    # global_scores = torch.tensor(self.prob_gmms[concept_idx].score_samples(c_sem[:, concept_idx:concept_idx+1].detach().cpu().numpy())).to(c_sem.device).unsqueeze(-1).unsqueeze(-1)
                    log_likelihoods = self.prob_gmms[concept_idx](c_sem[:, concept_idx:concept_idx+1]).unsqueeze(-1).unsqueeze(-1)
                    # mixed = (
                    #     dynamic_contexts[:, concept_idx, :self.emb_size] * c_sem[:, concept_idx:concept_idx+1] +
                    #     (1 - c_sem[:, concept_idx:concept_idx+1])*dynamic_contexts[:, concept_idx, self.emb_size:]
                    # )
                    # log_likelihoods = self.prob_gmms[concept_idx](mixed).unsqueeze(-1).unsqueeze(-1)
                    combined_log_likelihoods.append(log_likelihoods.squeeze(-1))
                    # global_selected = torch.where(
                    #     elf.beta_thresholds[concept_idx] > log_likelihoods,
                    #     torch.ones((log_likelihoods.shape[0], 1, 1)).to(log_likelihoods.device),
                    #     torch.zeros((log_likelihoods.shape[0], 1, 1)).to(log_likelihoods.device),
                    # )
                    # print("global_selected.shape =", global_selected.shape)
                    global_selected = self.sig(
                        self.temperature * (self.beta_thresholds[concept_idx] - log_likelihoods)
                    )
                    if not getattr(self, 'trained_gmms', False):
                        global_selected = torch.ones_like(global_selected)
                    if concept_idx == 0: print("\tTotal global selected embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples with a max value of ", torch.max(log_likelihoods), "and a min value of", torch.min(log_likelihoods), "when log thresh =", self.beta_thresholds[concept_idx].detach().cpu().numpy(), "and exp thresh =", torch.exp(self.beta_thresholds[concept_idx]).detach().cpu().numpy())
                    if concept_idx == 0: print("\t\tMax value of mask", torch.max(global_selected), "and a min value", torch.min(global_selected))
                    # if concept_idx == 0 and not training: print("\t\tMean of scores", torch.mean(log_likelihoods).detach().cpu().numpy(), "and std", torch.std(log_likelihoods).detach().cpu().numpy())
                    if concept_idx == 0: print("\tLearnt mean is:", self.prob_gmms[concept_idx].means.detach().cpu().numpy())
                    if concept_idx == 0: print("\tLearnt variances are:", torch.exp(self.prob_gmms[concept_idx].log_diag_covariances).detach().cpu().numpy())
                    # if concept_idx == 0: print("\tFirst 10 probs are:", c_sem[:, concept_idx][:10])
                    # if concept_idx == 0: print("\tFirst 10 log-likelihoods are:", log_likelihoods[:, concept_idx][:10])
                    # if concept_idx == 0: print("\tFirst 10 likelihoods are:", torch.exp(log_likelihoods)[:, concept_idx][:10])

                    pos_global_selected = global_selected
                    neg_global_selected = global_selected
                    if not training:
                        global_selected = torch.where(
                            global_selected >= 0.5,
                            torch.ones_like(global_selected),
                            torch.zeros_like(global_selected),
                        )
                # elif self.mode == 'ood_beta' and (self.prob_gmms is not None):
                #     # global_scores = torch.tensor(self.prob_gmms[concept_idx].score_samples(c_sem[:, concept_idx:concept_idx+1].detach().cpu().numpy())).to(c_sem.device).unsqueeze(-1).unsqueeze(-1)
                #     log_likelihoods = torch.tensor(self.prob_gmms[concept_idx].score_samples(c_logits[:, concept_idx:concept_idx+1].detach().cpu().numpy())).to(c_sem.device).unsqueeze(-1).unsqueeze(-1)
                #     combined_log_likelihoods.append(combined_log_likelihoods.squeeze(-1))
                #     global_selected = torch.where(
                #         self.prob_threshs[concept_idx] > log_likelihoods,
                #         torch.ones((log_likelihoods.shape[0], 1, 1)).to(log_likelihoods.device),
                #         torch.zeros((log_likelihoods.shape[0], 1, 1)).to(log_likelihoods.device),
                #     )
                #     # print("global_selected.shape =", global_selected.shape)
                #     # global_selected = self.sig(
                #     #     self.temperature * (log_likelihoods - self.beta_thresholds[concept_idx])
                #     # ).unsqueeze(-1).unsqueeze(-1)
                #     if concept_idx == 0 and not training: print("\tTotal global selected embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples with a max value of ", torch.max(log_likelihoods), "and a min value of", torch.min(log_likelihoods), "when self.beta_thresholds[concept_idx] =", self.beta_thresholds[concept_idx])
                #     if concept_idx == 0 and not training: print("\t\tMean of scores", torch.mean(log_likelihoods).detach().cpu().numpy(), "and std", torch.std(log_likelihoods).detach().cpu().numpy())
                #     pos_global_selected = global_selected
                #     neg_global_selected = global_selected
                #     if not training:
                #         global_selected = torch.where(
                #             global_selected >= 0.5,
                #             torch.ones_like(global_selected),
                #             torch.zeros_like(global_selected),
                #         )
                # elif self.mode == 'ood_beta':
                #     global_scores = self.log_beta_likelihood(c_sem, concept_idx=concept_idx).unsqueeze(-1).unsqueeze(-1)
                #     # print("global_scores.shape =", global_scores.shape)
                #     global_selected = torch.where(
                #         self.beta_thresholds[concept_idx] > global_scores,
                #         torch.ones((global_scores.shape[0], 1, 1)).to(global_scores.device),
                #         torch.zeros((global_scores.shape[0], 1, 1)).to(global_scores.device),
                #     )
                #     # print("global_selected.shape =", global_selected.shape)
                #     # global_selected = self.sig(
                #     #     self.temperature * (global_scores - self.beta_thresholds[concept_idx])
                #     # ).unsqueeze(-1).unsqueeze(-1)
                #     if concept_idx == 0 and not training: print("\tTotal global selected embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples with a max value of ", torch.max(global_scores), "and a min value of", torch.min(global_scores), "when self.beta_thresholds[concept_idx] =", self.beta_thresholds[concept_idx])
                #     if concept_idx == 0 and not training: print("\t\tMean of scores", torch.mean(global_scores).detach().cpu().numpy(), "and std", torch.std(global_scores).detach().cpu().numpy())
                #     pos_global_selected = global_selected
                #     neg_global_selected = global_selected
                #     if not training:
                #         global_selected = torch.where(
                #             global_selected >= 0.5,
                #             torch.ones_like(global_selected),
                #             torch.zeros_like(global_selected),
                #         )
                else:
                    pos_global_selected = torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device)
                    neg_global_selected = torch.zeros((dynamic_contexts.shape[0], 1, 1)).to(dynamic_contexts.device)


                pos_embeddings.append(
                    pos_global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
                    (1 - pos_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size]
                )
                neg_embeddings.append(
                    neg_global_selected * global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
                    (1 - neg_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:]
                )

            pos_embeddings = torch.concat(pos_embeddings, dim=1)
            neg_embeddings = torch.concat(neg_embeddings, dim=1)
            if training and combined_log_likelihoods:
                self._combined_log_likelihoods = torch.concat(combined_log_likelihoods, dim=-1)


        elif self.mode == 'ood_fixed':
            # Then we will generate a mixture of the dynamic and global
            # embedding weighted by the OOD weight
            pos_embeddings = []
            neg_embeddings = []
            selected = []
            for concept_idx in range(self.n_concepts):
                if training:
                    global_selected = torch.bernoulli(
                        torch.ones((dynamic_contexts.shape[0], 1)).to(dynamic_contexts.device) * self.ood_dropout_prob,
                    )
                else:
                    threshs = self.ood_thresholds[concept_idx:concept_idx+1].unsqueeze(-1).expand(dynamic_contexts.shape[0], -1)
                    scores = self._gmm_log_likelihood(
                        (
                            dynamic_contexts[:, concept_idx, :self.emb_size] * c_sem[:, concept_idx:concept_idx+1] +
                            (1 - c_sem[:, concept_idx:concept_idx+1])*dynamic_contexts[:, concept_idx, self.emb_size:]
                        ),
                        concept_idx=concept_idx
                    )
                    global_selected = torch.where(
                        self.gmm_thresholds[concept_idx] > scores,
                        torch.ones((dynamic_contexts.shape[0], 1)).to(dynamic_contexts.device),
                        torch.zeros((dynamic_contexts.shape[0], 1)).to(dynamic_contexts.device),
                    )
                # print("\tSelected global embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples with a max value of ", torch.max(global_selected), "and a min value of", torch.min(global_selected), "when self.ood_dropout_prob =", self.ood_dropout_prob)


                if self.approx_prediction_mode == 'new':
                    pos_embeddings.append(torch.concat(
                        [
                            dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                            global_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                        ],
                        dim=-1,
                    ))
                    neg_embeddings.append(torch.concat(
                        [
                            dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
                            global_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
                        ],
                        dim=-1,
                    ))
                else:
                    global_selected = global_selected.unsqueeze(-1)

                    # At this point, we are ready to mix the output embeddings
                    pos_embeddings.append(
                        global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
                        (1 - global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size]
                    )
                    neg_embeddings.append(
                        global_selected * global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
                        (1 - global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:]
                    )

                selected.append(global_selected)
            self._global_selected = torch.concat(selected, dim=1)
            pos_embeddings = torch.concat(pos_embeddings, dim=1)
            neg_embeddings = torch.concat(neg_embeddings, dim=1)

        elif self.mode == 'ood':
            # Then we will generate a mixture of the dynamic and global
            # embedding weighted by the OOD weight
            pos_embeddings = []
            neg_embeddings = []
            total_distances = []
            selected = []
            for concept_idx in range(self.n_concepts):
                # Compute the distance between the latent space and the
                # prototype held by this concept
                # Distances will have shape (B, 1)
                distances = torch.norm(
                    dynamic_contexts[:, concept_idx, :] - self.ood_centroids[concept_idx:concept_idx+1, :].expand(dynamic_contexts.shape[0], -1),
                    p=2,
                    dim=-1,
                ).unsqueeze(-1)
                total_distances.append(distances)
                # if concept_idx == 0: print("For concept", concept_idx)

                # if concept_idx == 0: print("\tMax distance is", torch.max(distances), "and min dinstance is", torch.mean(distances))

                # See if this distance is greater than the threshold for this
                # concept
                threshs = self.ood_thresholds[concept_idx:concept_idx+1].unsqueeze(-1).expand(dynamic_contexts.shape[0], -1)
                # if concept_idx == 0: print("\tThreshold is",  self.ood_thresholds[concept_idx])
                if self.log_thresholds:
                    threshs = torch.exp(threshs)
                # if concept_idx == 0: print("\tAfter exp, threshold is",  threshs[0, 0])
                diff = distances - threshs

                # Now do this as a differientable step function
                global_selected = self.sig(
                    # self.ood_fn_scales[concept_idx] * diff
                    self.temperature * diff
                )
                ###############################################################
                # if concept_idx == 0: print("\tBefore droput selected global embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples with a max value of ", torch.max(global_selected), "and a min value of", torch.min(global_selected))
                # if concept_idx == 0: print("\t\tThreshold is", torch.exp(self.ood_thresholds[concept_idx]))
                # if concept_idx == 0: print("\t\tMax distance is", torch.max(distances).detach().cpu().numpy())
                # if concept_idx == 0: print("\t\tMin distance is", torch.min(distances).detach().cpu().numpy())
                # if concept_idx == 0: print("\t\tPredicted prob is", c_sem[0, concept_idx].detach().cpu().numpy())
                # if concept_idx == 0 and hasattr(self, 'gmm'): print("\t\tBelow the GMM threshold:", np.sum(self.gmm[concept_idx].score_samples((dynamic_contexts[:, concept_idx, :self.emb_size]*c_sem[:, concept_idx:concept_idx+1] + (1 - c_sem[:, concept_idx:concept_idx+1])*dynamic_contexts[:, concept_idx, self.emb_size:]).detach().cpu().numpy()) < self.gmm_thresholds[concept_idx]))
                # New attempt!

                new_scores = self._gmm_log_likelihood(
                    (
                        dynamic_contexts[:, concept_idx, :self.emb_size] * c_sem[:, concept_idx:concept_idx+1] +
                        (1 - c_sem[:, concept_idx:concept_idx+1])*dynamic_contexts[:, concept_idx, self.emb_size:]
                    ),
                    concept_idx=concept_idx
                )
                global_selected = self.sig(
                    self.ood_fn_scales[concept_idx] * (self.gmm_thresholds[concept_idx] - new_scores)
                ).unsqueeze(-1)
                # if concept_idx == 0: print("\tBefore droput selected global embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples with a max value of ", torch.max(global_selected), "a min value of", torch.min(global_selected), ", and a mean value of", torch.mean(global_selected))
                # if concept_idx == 0: print("\t\tMax score", torch.max(new_scores).detach().cpu().numpy(), "and min score", torch.min(new_scores).detach().cpu().numpy())
                # if concept_idx == 0 and hasattr(self, 'gmm'): print("\t\tBelow the GMM threshold:", np.sum(new_scores.detach().cpu().numpy() < self.gmm_thresholds[concept_idx]))
                # if concept_idx == 0: print("\t\tself.ood_fn_scales[concept_idx] =", self.ood_fn_scales[concept_idx])
                # if concept_idx == 0: print("\t\tself.gmm_thresholds[concept_idx] =", self.gmm_thresholds[concept_idx])
                if training:
                    global_selected = torch.where(
                        global_selected > 0.5,
                        torch.ones_like(global_selected),
                        torch.zeros_like(global_selected),
                    )

                ################################################################

                # Let's do any dropout if necessary
                if training:
                    forced_on = self.ood_dropout(torch.ones_like(global_selected))
                    global_selected = torch.ones_like(global_selected) * forced_on + (1 - forced_on) * global_selected
                    # if concept_idx == 0: print("\tAfter droput selected global embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples")
                # if concept_idx == 0: print("\t\tAfter droput selected global embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples with a max value of ", torch.max(global_selected), "a min value of", torch.min(global_selected), ", and a mean value of", torch.mean(global_selected))


                if self.approx_prediction_mode == 'new':
                    pos_embeddings.append(torch.concat(
                        [
                            dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                            global_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                        ],
                        dim=-1,
                    ))
                    neg_embeddings.append(torch.concat(
                        [
                            dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
                            global_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
                        ],
                        dim=-1,
                    ))
                else:
                    # if concept_idx == 0: print("\tSelected global embedding for", torch.sum(global_selected > 0.5).detach().cpu().numpy(), "out of", global_selected.shape[0], "samples")
                    global_selected = global_selected.unsqueeze(-1)

                    # At this point, we are ready to mix the output embeddings
                    pos_embeddings.append(
                        global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
                        (1 - global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size]
                    )
                    neg_embeddings.append(
                        global_selected * global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
                        (1 - global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:]
                    )

                selected.append(global_selected)
            self._global_selected = torch.concat(selected, dim=1)
            pos_embeddings = torch.concat(pos_embeddings, dim=1)
            neg_embeddings = torch.concat(neg_embeddings, dim=1)
            self._distances = torch.concat(total_distances, dim=1)
        else:
            raise ValueError(
                f'Unsupported mode "{self.mode}"'
            )


        return c_sem, pos_embeddings, neg_embeddings, extra_outputs



    def _extra_losses(
        self,
        x,
        y,
        c,
        y_pred,
        c_sem,
        c_pred,
        competencies=None,
        prev_interventions=None,
    ):
        loss = 0.0
        if (
            (self._global_contexts is not None) and
            (self._dynamic_contexts is not None) and
            (self.l2_dist_loss_weight != 0) and (
                self.mode == 'approx'
            )
        ):
            loss += self.l2_dist_loss_weight * torch.norm(
                self._dynamic_contexts - self._global_contexts,
                p=2,
            ).mean()

            # And reset stuff
            self._dynamic_contexts = None
            self._global_contexts = None
        if self.distance_l2_loss and (self._distances is not None):
            mean_distances = torch.mean(
                self._distances
            )
            loss += self.distance_l2_loss * mean_distances
            self._distances = None
        if self.thresh_l2_loss and self.mode == 'ood':
            loss += self.thresh_l2_loss * (
                torch.norm(self.ood_thresholds, p=2)/self.n_concepts
            )

        if self.mode == 'ood_beta' and (
            self._combined_log_likelihoods is not None
        ) and (self.ood_loss_weight):
            # print("self._combined_log_likelihoods.shape =", self._combined_log_likelihoods.shape)
            loss += self.ood_loss_weight * torch.mean(
                (-self._combined_log_likelihoods).mean(-1)
            )
            if getattr(self, 'trained_gmms', False):
                loss += -0.001 * torch.mean(self.beta_thresholds)
            self._combined_log_likelihoods = None

        return loss


    def freeze_non_approx_weights(
        self,
        freeze_label_predictor=True,
        freeze_concept_rank_model=True,
    ):
        if freeze_label_predictor:
            for param in self.c2y_model.parameters():
                param.requires_grad = False
            for param in self.c2y_model_approx.parameters():
                param.requires_grad = False
        for param in self.pre_concept_model.parameters():
            param.requires_grad = False
        for emb_generator in self.concept_context_generators:
            for param in emb_generator.parameters():
                param.requires_grad = False
        for emb_generator in self.concept_prob_generators:
            for param in emb_generator.parameters():
                param.requires_grad = False
        if freeze_concept_rank_model:
            for param in self.concept_rank_model.parameters():
                param.requires_grad = False


    def unfreeze_non_approx_weights(self):
        for param in self.c2y_model.parameters():
            param.requires_grad = True
        for param in self.c2y_model_approx.parameters():
            param.requires_grad = True
        for param in self.pre_concept_model.parameters():
            param.requires_grad = True
        for emb_generator in self.concept_context_generators:
            for param in emb_generator.parameters():
                param.requires_grad = True
        for emb_generator in self.concept_prob_generators:
            for param in emb_generator.parameters():
                param.requires_grad = True
        for param in self.concept_rank_model.parameters():
            param.requires_grad = True

    def freeze_non_ood_weights(
        self,
        freeze_label_predictor=True,
        freeze_concept_rank_model=True,
        freeze_global_concept_generators=True,
    ):
        self.freeze_non_approx_weights(
            freeze_label_predictor=freeze_label_predictor,
            freeze_concept_rank_model=freeze_concept_rank_model,
        )
        if freeze_global_concept_generators:
            self.concept_embeddings.requires_grad = False
            for submodule in self.attn_model:
                if hasattr(submodule, 'parameters'):
                    for param in submodule.parameters():
                        param.requires_grad = False

    def unfreeze_non_ood_weights(self):
        self.unfreeze_non_approx_weights()
        self.concept_embeddings.requires_grad = True
        for submodule in self.attn_model:
            if hasattr(submodule, 'parameters'):
                for param in submodule.parameters():
                    param.requires_grad = True
