import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch

from torchvision.models import resnet50

from cem.models.cbm import ConceptBottleneckModel, compute_accuracy
from cem.models.cem import ConceptEmbeddingModel
import cem.train.utils as utils
from cem.models.intcbm import IntAwareConceptEmbeddingModel


def log(x):
    return torch.log(x + 1e-6)

def Gaussian_CDF(x): #change from erf function
    return 0.5 * (1. + torch.erf(x / np.sqrt(2.)))


def _binary_entropy(probs):
    return -probs * torch.log2(probs) - (1 - probs) * torch.log2(1 - probs)




class IntAwareMixCEM(IntAwareConceptEmbeddingModel):
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
        task_loss_weight=0,
        intervention_task_loss_weight=1,

        # New args
        initial_concept_embeddings=None,
        fixed_embeddings=False,
        residual_scale=None,
        learnable_residual_scale=True,
        residual_scale_reg=0,
        sigmoidal_residual_scale=False,
        sigmoidal_residual=False,
        residual_scale_norm_metric=1,
        normalize_residual=False,
        residual_nll_reg=0,
        fixed_residual_scale=None,
        intermediate_task_concept_loss=0,
        drop_residual_prob=0,
        scalar_residual=False,
        use_distance_probs=False,
    ):
        assert task_loss_weight == 0, (
            f'IntCEM only supports task_loss_weight=0 as this loss is included '
            f'as part of the trajectory loss. It was given task_loss_weight = '
            f'{task_loss_weight}'
        )
        self.num_rollouts = num_rollouts
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map
        if len(concept_map) == n_concepts:
            use_concept_groups = False

        ConceptEmbeddingModel.__init__(
            self,
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            shared_prob_gen=False,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            c_extractor_arch=c_extractor_arch,
            output_latent=output_latent,
            optimizer=optimizer,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            top_k_accuracy=top_k_accuracy,
            use_concept_groups=use_concept_groups,
            context_gen_out_size=emb_size if use_distance_probs else 2 * emb_size,
        )
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map

        self.emb_size = emb_size

        units = [
            n_concepts * emb_size + # Bottleneck
            n_concepts # Prev interventions
        ] + (int_model_layers or [256, 128]) + [
            len(self.concept_map) if self.use_concept_groups else n_concepts
        ]
        layers = []
        for i in range(1, len(units)):
            if int_model_use_bn:
                layers.append(
                    torch.nn.BatchNorm1d(num_features=units[i-1]),
                )
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())
        self.concept_rank_model = torch.nn.Sequential(*layers)

        self.intervention_discount = intervention_discount
        self.intervention_task_discount = intervention_task_discount
        self.horizon_rate = horizon_rate
        self.horizon_limit = torch.nn.Parameter(
            torch.FloatTensor([initial_horizon]),
            requires_grad=False,
        )
        self.current_steps = torch.nn.Parameter(
            torch.IntTensor([0]),
            requires_grad=False,
        )
        self.rollout_init_steps = rollout_init_steps
        self.intervention_weight = intervention_weight
        self.loss_interventions = torch.nn.CrossEntropyLoss()
        self.max_horizon = max_horizon
        self.include_only_last_trajectory_loss = \
            include_only_last_trajectory_loss
        self.intervention_task_loss_weight = intervention_task_loss_weight
        self.use_concept_groups = use_concept_groups
        self._horizon_distr = lambda init, end: np.random.randint(
            init,
            end,
        )

        ###########################################################################
        # NEW STUFF

        if (initial_concept_embeddings is False) or (
            initial_concept_embeddings is None
        ):
            initial_concept_embeddings = torch.rand(
                self.n_concepts,
                2,
                emb_size,
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


        if residual_scale is not None:
            self.residual_scale = torch.nn.Parameter(
                residual_scale * torch.ones((self.n_concepts,)),
                requires_grad=learnable_residual_scale,
            )
        else:
            self.residual_scale = torch.nn.Parameter(
                torch.rand((self.n_concepts,)),
                requires_grad=learnable_residual_scale,
            )

        self.residual_scale_reg = residual_scale_reg
        self.sigmoidal_residual_scale = sigmoidal_residual_scale
        self.sigmoidal_residual = sigmoidal_residual
        self.residual_scale_norm_metric = residual_scale_norm_metric
        self.normalize_residual = normalize_residual
        self.drop_residual_prob = drop_residual_prob
        self.scalar_residual = scalar_residual
        self.norm_distr = torch.distributions.Normal(
            torch.zeros((self.emb_size,)),
            torch.ones((self.emb_size,)),
        )

        self.intermediate_task_concept_loss = intermediate_task_concept_loss
        self._no_residual_contexts = None

        # Residual distribution
        self.nll_loss = torch.nn.GaussianNLLLoss()
        self.residual_nll_reg = residual_nll_reg
        self._residuals = None
        self.residual_means = torch.nn.Parameter(
            torch.rand((self.n_concepts, self.emb_size)),
            requires_grad=True,
        )
        if fixed_residual_scale:
            self.residual_logscales = torch.nn.Parameter(
                torch.log(fixed_residual_scale) * torch.ones((self.n_concepts,)),
                requires_grad=False,
            )
        else:
            self.residual_logscales = torch.nn.Parameter(
                torch.rand((self.n_concepts,)),
                requires_grad=True,
            )
        self.use_distance_probs = use_distance_probs
        if use_distance_probs:
            self.contrastive_scale = torch.nn.Parameter(
                torch.rand((self.n_concepts,)),
                requires_grad=True,
            )

    def _compute_task_loss(
        self,
        y,
        probs=None,
        pos_embeddings=None,
        neg_embeddings=None,
        y_pred_logits=None,
    ):
        if y_pred_logits is not None:
            loss_with_residual = self.loss_task(
                (
                    y_pred_logits if y_pred_logits.shape[-1] > 1
                    else y_pred_logits.reshape(-1)
                ),
                y,
            )
        else:
            bottleneck = (
                pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                    neg_embeddings * (
                        1 - torch.unsqueeze(probs, dim=-1)
                    )
                )
            )
            bottleneck = bottleneck.view(
                (-1, self.emb_size * self.n_concepts)
            )
            y_logits = self.c2y_model(bottleneck)
            loss_with_residual = self.loss_task(
                (
                    y_logits
                    if y_logits.shape[-1] > 1 else
                    y_logits.reshape(-1)
                ),
                y,
            )

        if self.intermediate_task_concept_loss:
            pos_global_embs = self.concept_embeddings[:, 0, :].expand(probs.shape[0], -1, -1)
            neg_global_embs = self.concept_embeddings[:, 1, :].expand(probs.shape[0], -1, -1)

            no_residual_bottleneck = (
                pos_global_embs * torch.unsqueeze(probs, dim=-1) + (
                    neg_global_embs * (
                        1 - torch.unsqueeze(probs, dim=-1)
                    )
                )
            )

            no_residual_bottleneck = no_residual_bottleneck.view(
                (-1, self.emb_size * self.n_concepts)
            )
            no_residual_y_logits = self.c2y_model(no_residual_bottleneck)
            loss_with_residual = self.loss_task(
                (
                    no_residual_y_logits
                    if no_residual_y_logits.shape[-1] > 1 else
                    no_residual_y_logits.reshape(-1)
                ),
                y,
            )
            loss_without_residual = self.intermediate_task_concept_loss * loss_with_residual
        else:
            loss_without_residual = 0
        return loss_with_residual + loss_without_residual

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
        if self.residual_scale_reg:
            if self.sigmoidal_residual_scale:
                loss += torch.norm(self.sig(self.residual_scale), p=self.residual_scale_norm_metric)/self.n_concepts
            else:
                loss += torch.norm(self.residual_scale, p=self.residual_scale_norm_metric)/self.n_concepts

        scales = torch.exp(self.residual_logscales)

        if self.residual_nll_reg and (self._residuals is not None):
            for i in range(self.n_concepts):
                background = self.norm_distr.sample()
                ref_distr = (self.residual_means[i, :] + background) * scales[i]
                nll = self.nll_loss(
                    input=self._residuals[:, i, :],
                    target=ref_distr,
                )
                loss += (self.residual_nll_reg * nll)/self.n_concepts
            self._residuals = None

        # if self.intermediate_task_concept_loss and (
        #     self._no_residual_contexts is not None
        # ):
        #     no_residual_pos_embeddings = self._no_residual_contexts[:, :, :self.emb_size]
        #     no_residual_neg_embeddings = self._no_residual_contexts[:, :, self.emb_size:]
        #     mixed_no_residual = no_residual_pos_embeddings * c_sem.unsqueeze(-1) - (1 - c_sem.unsqueeze(-1)) * no_residual_neg_embeddings
        #     y_pred_logits = self.c2y_model(mixed_no_residual.view(mixed_no_residual.shape[0], -1))
        #     loss += self.intermediate_task_concept_loss * self.loss_task(
        #         (
        #             y_pred_logits if y_pred_logits.shape[-1] > 1
        #             else y_pred_logits.reshape(-1)
        #         ),
        #         y,
        #     )
            self._no_residual_contexts = None

        return loss

    def _relaxed_multi_bernoulli_sample(self, probs, temperature=1, idx=None):
        # Sample from a standard Gaussian first to perform the
        # reparameterization trick
        if len(probs.shape):
            shape = (probs.shapea[0],)
        else:
            shape = []
        epsilon = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)).to(
            probs.device
        )
        u = Gaussian_CDF(epsilon)
        return torch.sigmoid(
            1.0/temperature * (
                log(probs) - log(1. - probs) + log(u) - log(1. - u)
            )
        )

    def _distance_metric(self, neg_anchor, pos_anchor, latent):
        neg_dist = (neg_anchor - latent).pow(2).sum(-1).sqrt()
        pos_dist = (pos_anchor - latent).pow(2).sum(-1).sqrt()
        return neg_dist - pos_dist

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if training:
            self._residuals = None
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            # if training and self.intermediate_task_concept_loss:
            #     self._no_residual_contexts = []
            c_sem = []
            residuals = []
            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.use_distance_probs:
                    latent_space = context_gen(pre_c)
                    # [Shape: (1, emb_size)]
                    anchor_concept_pos_emb = torch.unsqueeze(
                        self.concept_embeddings[i, 0, :],
                        dim=0,
                    )
                    # [Shape: (1, emb_size)]
                    anchor_concept_neg_emb = torch.unsqueeze(
                        self.concept_embeddings[i, 1, :],
                        dim=0,
                    )
                    # [Shape: (B)]
                    prob = self.sig(
                        self.contrastive_scale[i] * self._distance_metric(
                            neg_anchor=anchor_concept_neg_emb,
                            pos_anchor=anchor_concept_pos_emb,
                            latent=latent_space,
                        )
                    )
                    # [Shape: (B, 1)]
                    prob = torch.unsqueeze(prob, dim=-1)
                    latent_space = prob * anchor_concept_pos_emb + (1 - prob) * anchor_concept_neg_emb
                    context = torch.concat(
                        [
                            anchor_concept_pos_emb.expand(prob.shape[0], -1),
                            anchor_concept_neg_emb.expand(prob.shape[0], -1)
                        ],
                        dim=-1,
                    )
                    contexts.append(torch.unsqueeze(context, dim=1))
                    c_sem.append(prob)
                else:
                    if self.shared_prob_gen:
                        prob_gen = self.concept_prob_generators[0]
                    else:
                        prob_gen = self.concept_prob_generators[i]
                    residual = context_gen(pre_c)
                    if self.sigmoidal_residual:
                        residual = self.sig(residual)
                    # Shape: (B, 2*emb_size)
                    global_embs = self.concept_embeddings[i:i+1, :, :].view(1, -1).expand(residual.shape[0], -1)
                    scale = self.residual_scale[i]
                    if self.sigmoidal_residual_scale:
                        scale = self.sig(scale)
                    if (not self.scalar_residual) and self.normalize_residual:
                        residual = torch.nn.functional.normalize(residual.view(residual.shape[0], 2, self.emb_size), dim=-1).view(residual.shape[0], -1)
                    residuals.append(residual.unsqueeze(1))
                    # if training and self.intermediate_task_concept_loss:
                    #     self._no_residual_contexts.append(torch.unsqueeze(global_embs, dim=1))
                    if self.scalar_residual:
                        # Turn the residual from (B, 2) to (B, 2*m) by repeating the
                        # element across the embedding that it corresponds to
                        # residual = torch.concat(
                        #     [
                        #         residual[:, 0:1].expand(-1, self.emb_size),
                        #         residual[:, 1:].expand(-1, self.emb_size),
                        #     ],
                        #     dim=-1,
                        # )
                        residual = torch.exp(residual)
                        if self.drop_residual_prob > 0:
                            mask = np.random.binomial(
                                n=1,
                                p=self.drop_residual_prob,
                                size=residual.shape[0],
                            )
                            mask = torch.tensor(mask).to(residual.device).unsqueeze(-1)
                            context = scale * residual * (mask - 1) * global_embs + mask * global_embs
                        else:
                            context = scale * residual * global_embs
                    else:
                        if self.drop_residual_prob > 0:
                            mask = np.random.binomial(
                                n=1,
                                p=self.drop_residual_prob,
                                size=residual.shape[0],
                            )
                            mask = torch.tensor(mask).to(residual.device).unsqueeze(-1)
                            context = scale * residual * (mask - 1) + global_embs
                        else:
                            context = scale * residual + global_embs
                    prob = prob_gen(context)
                    contexts.append(torch.unsqueeze(context, dim=1))
                    c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, dim=-1)
            contexts = torch.cat(contexts, dim=1)
            if len(residuals):
                residuals = torch.cat(residuals, dim=1)
            # if training and self.intermediate_task_concept_loss:
            #     self._no_residual_contexts = torch.cat(self._no_residual_contexts, dim=1)
            latent = contexts, c_sem, residuals
        else:
            contexts, c_sem, residuals = latent
        if training:
            self._residuals = residuals
        pos_embeddings = contexts[:, :, :self.emb_size]
        neg_embeddings = contexts[:, :, self.emb_size:]
        return c_sem, pos_embeddings, neg_embeddings, {
            # 'no_residual_pos_embeddings': no_residual_contexts[:, :, :self.emb_size],
            # 'no_residual_neg_embeddings': no_residual_contexts[:, :, self.emb_size:],
        }