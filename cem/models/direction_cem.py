import numpy as np
import pytorch_lightning as pl
import scipy
import sklearn.metrics
import torch
import math

from torchvision.models import resnet50
from cem.models.intcbm import IntAwareConceptEmbeddingModel
import cem.train.utils as utils
from cem.models.cbm import compute_accuracy



def dot(x, y):
    return (x * y).sum(dim=-1).unsqueeze(-1)


class DynamicDropoutLayer(torch.nn.Dropout):

    def __init__(
        self,
        *args,
        rate=1,
        min_prob=0,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.training_steps = torch.nn.Parameter(
            torch.IntTensor([0]),
            requires_grad=False,
        )
        self.min_prob = min_prob
        self.rate = rate

    def forward(self, input):
        if self.training:
            self.training_steps += 1
            self.p = max(self.p * self.rate, self.min_prob)
        return super().forward(input)


class ProjectionConceptEmbeddingModel(IntAwareConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=128,
        training_intervention_prob=0.25,
        embedding_activation=None,
        concept_loss_weight=1,
        task_loss_weight=1,
        intermediate_task_concept_loss=0,
        intervention_task_discount=1,

        # Mixing
        bottleneck_pooling='concat',

        fixed_embeddings=False,
        initial_concept_embeddings=None,
        use_cosine_similarity=False,
        fixed_scale=None,

        # Extra capacity
        extra_capacity=0,
        extra_capacity_dropout_prob=0,

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
        tau=1,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        top_k_accuracy=None,


        warmup_mode=False,
        include_bypass_model=False,
        simplified_mode=False,

        shared_emb_generator=True,
        use_triplet_loss=False,
        use_learnable_prob=False,
        learnable_orthogonal_dir=0,
        single_residual_vector=False,
        sigmoidal_extra_capacity=True,
        conditional_residual=True,
        mix_residuals=False,
        residual_sep_loss=0,
        manual_residual_scale=1,
        residual_ood_detection=1,

        # INTCEM ARGS
        rollout_init_steps=0,
        int_model_layers=None,
        int_model_use_bn=True,
        num_rollouts=1,
        intervention_discount=1,
        include_only_last_trajectory_loss=True,
        intervention_task_loss_weight=1,
        intervention_weight=5,
        concept_map=None,
        max_horizon=6,
        initial_horizon=2,
        horizon_rate=1.005,

        # Adversarial stuff
        adversary_loss_weight=0,
        warmup_period=0,
        use_learnable_residual=False,
        dyn_scaling=100,
        residual_weight_l2=0,
        residual_drop_prob=0,

        **kwargs,
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.residual_weight_l2 = residual_weight_l2
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        if bottleneck_pooling == 'pcm':
            bottleneck_pooling = 'per_class_mixing'
        self.bottleneck_pooling = bottleneck_pooling
        if single_residual_vector:
            extra_capacity = extra_capacity or emb_size
        self.extra_capacity = extra_capacity
        if extra_capacity and not (single_residual_vector):
            assert bottleneck_pooling == 'concat', 'Currently only support extra capcity when using concat pooling'
            self._extra_capacity_residual =  torch.nn.Sequential(*[
                torch.nn.LeakyReLU(),
                torch.nn.Linear(emb_size, self.extra_capacity),
                torch.nn.Dropout(p=extra_capacity_dropout_prob),
                torch.nn.Sigmoid(),
            ])
            self.extra_capacity_residual = lambda x, *args: self._extra_capacity_residual(x)
        else:
            self.extra_capacity_residual = None
        assert self.bottleneck_pooling in ['concat', 'additive', 'mean', 'weighted_mean', 'per_class_mixing'], (
            f'We only support bottleneck pooling functions "concat", '
            f'"additive", "per_class_mixing", "weighted_mean", and "mean". However we were given '
            f'pooling function "{bottleneck_pooling}".'
        )
        if self.bottleneck_pooling == 'weighted_mean':
            self.concept_pool_weights = torch.nn.Parameter(
                torch.rand((self.n_concepts,)),
                requires_grad=True,
            )
        if self.bottleneck_pooling in ["additive", "mean", "weighted_mean"]:
            self.bottleneck_size = emb_size + self.extra_capacity
        else:
            if single_residual_vector:
                self.bottleneck_size = emb_size * n_concepts + self.extra_capacity
            else:
                self.bottleneck_size = 2*emb_size * n_concepts + self.extra_capacity
        if c_extractor_arch == "identity":
            self.pre_concept_model = lambda x: x
        else:
            self.pre_concept_model = c_extractor_arch(output_dim=None)
        if self.pre_concept_model == "identity":
            c_extractor_arch = "identity"
            self.pre_concept_model = lambda x: x


        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        self.intervention_task_discount = intervention_task_discount
        self.intermediate_task_concept_loss = intermediate_task_concept_loss

        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.top_k_accuracy = top_k_accuracy

        self.use_triplet_loss = use_triplet_loss
        self._manual_residual_scale = manual_residual_scale
        if (initial_concept_embeddings is False) or (
            initial_concept_embeddings is None
        ):
            if self.use_triplet_loss:
                initial_concept_embeddings = torch.rand(
                    self.n_concepts,
                    2,
                    emb_size,
                )
            else:
                initial_concept_embeddings = torch.rand(
                    self.n_concepts,
                    emb_size,
                )
        else:
            if isinstance(initial_concept_embeddings, np.ndarray):
                initial_concept_embeddings = torch.FloatTensor(
                    initial_concept_embeddings
                )
            emb_size = initial_concept_embeddings.shape[-1]
        self.emb_size = emb_size
        self.concept_embeddings = torch.nn.Parameter(
            initial_concept_embeddings,
            requires_grad=(not fixed_embeddings),
        )
        self.learnable_orthogonal_dir = learnable_orthogonal_dir
        if learnable_orthogonal_dir:
            self.orthogonal_embeddings = torch.nn.Parameter(
                torch.rand(
                    self.n_concepts,
                    2,
                    emb_size,
                ),
                requires_grad=True,
            )

            self.orthogonal_scale_models = torch.nn.ModuleList()
            for i in range(n_concepts):
                self.orthogonal_scale_models.append(
                    torch.nn.Sequential(*[
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            emb_size,
                            1,
                        ),
                        torch.nn.Sigmoid(),
                    ])
                )

        # if use_triplet_loss:
        #     if fixed_scale is not None:
        #         self.contrastive_scale_pos = torch.nn.Parameter(
        #             fixed_scale * torch.ones((self.n_concepts,)),
        #             requires_grad=False,
        #         )
        #         self.contrastive_scale_neg = torch.nn.Parameter(
        #             fixed_scale * torch.ones((self.n_concepts,)),
        #             requires_grad=False,
        #         )
        #     else:
        #         self.contrastive_scale_pos = torch.nn.Parameter(
        #             torch.rand((self.n_concepts,)),
        #             requires_grad=True,
        #         )
        #         self.contrastive_scale_neg = torch.nn.Parameter(
        #             torch.rand((self.n_concepts,)),
        #             requires_grad=True,
        #         )
        # else:
        if fixed_scale is not None:
            self.contrastive_scale = torch.nn.Parameter(
                fixed_scale * torch.ones((self.n_concepts,)),
                requires_grad=False,
            )
        else:
            self.contrastive_scale = torch.nn.Parameter(
                torch.rand((self.n_concepts,)),
                requires_grad=True,
            )

        self.concept_emb_generators = torch.nn.ModuleList()
        self.shared_emb_generator = shared_emb_generator
        for i in range(n_concepts):
            if c_extractor_arch == "identity":
                self.concept_emb_generators.append(torch.nn.Identity())
                continue
            if embedding_activation is None:
                emb_act = torch.nn.Identity()
            elif embedding_activation == "sigmoid":
                emb_act = torch.nn.Sigmoid()
            elif embedding_activation == "leakyrelu":
                emb_act = torch.nn.LeakyReLU()
            elif embedding_activation == "relu":
                emb_act = torch.nn.ReLU()
            else:
                raise ValueError(
                    f'Unsupported embedding activation "{embedding_activation}"'
                )
            n_in_feats = list(
                self.pre_concept_model.modules()
            )[-1].out_features
            if self.shared_emb_generator:
                if len(self.concept_emb_generators) == 0:
                    self.concept_emb_generators.append(
                        torch.nn.Sequential(*[
                            # torch.nn.BatchNorm1d(num_features=n_in_feats),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(
                                n_in_feats,
                                emb_size,
                            ),
                            # torch.nn.LeakyReLU(),
                            # torch.nn.Linear(
                            #     emb_size,
                            #     emb_size,
                            # ),
                            emb_act,
                        ])
                    )
                else:
                    self.concept_emb_generators.append(
                        self.concept_emb_generators[0]
                    )
            else:
                self.concept_emb_generators.append(
                    torch.nn.Sequential(*[
                        # torch.nn.BatchNorm1d(num_features=n_in_feats),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            n_in_feats,
                            emb_size,
                        ),
                        # torch.nn.LeakyReLU(),
                        # torch.nn.Linear(
                        #     emb_size,
                        #     emb_size,
                        # ),
                        emb_act,
                    ])
                )

        self.residual_masking = lambda x: x
        if residual_drop_prob:
            if isinstance(residual_drop_prob, (int, float)):
                self.residual_masking = DynamicDropoutLayer(p=residual_drop_prob, rate=1)
            elif isinstance(residual_drop_prob, str) and residual_drop_prob.startswith('dyn_'):
                rate, min_prob = residual_drop_prob.split("_")[1:]
                self.residual_masking = DynamicDropoutLayer(
                    p=1,
                    rate=float(rate),
                    min_prob=float(min_prob),
                )
            else:
                raise ValueError(f'Unsupported residual_drop_prob {residual_drop_prob}')


        if c2y_model is None:
            # Else we construct it here directly
            if 'per_class_mixing' in self.bottleneck_pooling:
                additional_dims = 0
                if single_residual_vector:
                    additional_dims = n_concepts if mix_residuals else 1
                else:
                    additional_dims = 0
                self.downstream_label_weights = torch.nn.Parameter(
                    torch.rand((n_tasks, self.n_concepts + additional_dims)),
                    requires_grad=True,
                )
                if self.bottleneck_pooling == 'per_class_mixing_shared':
                    self.class_embeddings = torch.nn.Parameter(
                        torch.rand((n_tasks, 2*self.emb_size)),
                        requires_grad=True,
                    )
                    for i in range(n_tasks):
                        units = [self.emb_size * 2] + (c2y_layers or []) + [1]
                        layers = []
                        for i in range(1, len(units)):
                            layers.append(torch.nn.LeakyReLU())
                            layers.append(torch.nn.Linear(units[i-1], units[i]))
                    self.c2y_model = torch.nn.Sequential(*layers)
                else:
                    # self.c2y_model = torch.nn.ModuleList()
                    # for i in range(n_tasks):
                    #     units = [self.emb_size] + (c2y_layers or []) + [1]
                    #     layers = []
                    #     for i in range(1, len(units)):
                    #         layers.append(torch.nn.Linear(units[i-1], units[i]))
                    #         if i != len(units) - 1:
                    #             layers.append(torch.nn.LeakyReLU())
                    #     self.c2y_model.append(torch.nn.Sequential(*layers))
                    units = [self.emb_size] + (c2y_layers or []) + [1]
                    layers = []
                    for i in range(1, len(units)):
                        layers.append(torch.nn.Linear(units[i-1], units[i]))
                        if i != len(units) - 1:
                            layers.append(torch.nn.LeakyReLU())
                    self.c2y_model = torch.nn.Sequential(*layers)
            else:
                units = [self.bottleneck_size] + (c2y_layers or []) + [n_tasks]
                layers = []
                for i in range(1, len(units)):
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                    if i != len(units) - 1:
                        layers.append(torch.nn.LeakyReLU())
                self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model

        self.sig = torch.nn.Sigmoid()
        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights, reduction=('none' if simplified_mode else 'mean'))
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights,
                reduction=('none' if simplified_mode else 'mean'),
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.n_tasks = n_tasks
        self.tau = tau
        self.use_concept_groups = use_concept_groups

        # OOD Detection
        self.residual_ood_detection = residual_ood_detection

        self._cos_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)


        self.warmup_mode = warmup_mode
        if self.warmup_mode or include_bypass_model:
            self.bypass_label_predictor = torch.nn.Linear(
                (
                    emb_size if shared_emb_generator else list(
                        self.pre_concept_model.modules()
                    )[-1].out_features
                ),
                n_tasks,
            )
        self.simplified_mode = simplified_mode

        self.single_residual_vector = single_residual_vector
        self.conditional_residual = conditional_residual
        self.mix_residuals = mix_residuals
        self.residual_sep_loss = residual_sep_loss
        self.residual_model = None
        self._residual = None
        self._residual_in_bn = None
        if single_residual_vector:
            if self.conditional_residual:
                self.mixture_coefficients = torch.nn.Parameter(
                    torch.rand(
                        self.n_concepts,
                    ),
                    requires_grad=True,
                )
            self._dropout = torch.nn.Dropout(p=extra_capacity_dropout_prob, inplace=False)
            def _orth_fn(x):
                latent_space = x[:, :self.emb_size]
                mixed_emb = x[:, self.emb_size:]
                y_y = dot(mixed_emb, mixed_emb)
                x_y = dot(latent_space, mixed_emb)
                return self._dropout(latent_space - (x_y * mixed_emb) / (y_y + 1e-16))
            if use_learnable_residual:
                n_in_feats = list(
                    self.pre_concept_model.modules()
                )[-1].out_features
                n_in_feats = n_in_feats + emb_size if self.conditional_residual else n_in_feats
                if self.mix_residuals:
                    if self.residual_sep_loss:
                        self.scoring_model = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(num_features=2*extra_capacity),
                            torch.nn.Linear(
                                2*extra_capacity,
                                1,
                            ),
                            torch.nn.Sigmoid(),
                        )
                    self._residual_in_bn = torch.nn.BatchNorm1d(num_features=n_in_feats)
                    self._residual_model = torch.nn.Sequential(
                        self._residual_in_bn,
                        torch.nn.Linear(
                            n_in_feats,
                            2*self.n_concepts*extra_capacity,
                        ),
                        torch.nn.Unflatten(-1, (self.n_concepts, 2, extra_capacity)),
                        torch.nn.Sigmoid() if sigmoidal_extra_capacity else torch.nn.Identity(),
                    )
                    self._residual_mix_coefficients = torch.nn.Parameter(
                        torch.rand(
                            self.n_concepts,
                        ),
                        requires_grad=True,
                    )
                    self.residual_model = self._residual_model
                else:
                    self._residual_in_bn = torch.nn.BatchNorm1d(num_features=n_in_feats)
                    self.residual_model = torch.nn.Sequential(
                        self._residual_in_bn,
                        torch.nn.Linear(
                            n_in_feats,
                            extra_capacity,
                        ),
                        torch.nn.Sigmoid() if sigmoidal_extra_capacity else torch.nn.Identity(),
                    )
            else:
                self.residual_model = _orth_fn



        # Adversarial!
        self.dyn_scaling = dyn_scaling
        self.adversary_loss_weight = adversary_loss_weight
        self.num_steps = torch.nn.Parameter(
            torch.IntTensor([0]),
            requires_grad=False,
        )
        self.warmup_period = warmup_period
        if adversary_loss_weight:
            # Then we will make an adversarial model here
            self.discriminator = torch.nn.Sequential(
                # torch.nn.Linear(self.extra_capacity, 128),
                # torch.nn.LeakyReLU(),
                # torch.nn.Linear(128, 64),
                # torch.nn.LeakyReLU(),
                torch.nn.Linear(self.extra_capacity, self.n_concepts),
                torch.nn.Sigmoid(),
            )
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)


        # INTCEM STUFF!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.num_rollouts = num_rollouts
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map

        # Else we construct it here directly
        units = [
            self.bottleneck_size + # Bottleneck
            n_concepts # Prev interventions
        ] + (int_model_layers or [256, 128]) + [
            len(self.concept_map) if self.use_concept_groups else n_concepts
        ]
        layers = [torch.nn.Flatten(start_dim=1, end_dim=-1)]
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
        self._task_loss_kwargs = {}

        self.use_learnable_prob = use_learnable_prob
        if not (self.use_triplet_loss or use_learnable_prob):
            # Compute the probability of activation based on the cosine
            # similarity
            self.prob_score_model = lambda concept_latent_space, pos_symbolic_vector, neg_symbolic_vector, pos_neural_vector, neg_neural_vector, idx: (self._cos_similarity(concept_latent_space, pos_symbolic_vector) + 1)/2
        elif use_learnable_prob:
            self._prob_score_model = torch.nn.Sequential(
                torch.nn.Linear(2*(emb_size + extra_capacity), 1),
                torch.nn.Sigmoid(),
            )
            self.prob_score_model = lambda concept_latent_space, pos_symbolic_vector, neg_symbolic_vector, pos_neural_vector, neg_neural_vector, idx: self._prob_score_model(
                torch.concat(
                    [pos_symbolic_vector, pos_neural_vector, neg_symbolic_vector, neg_neural_vector],
                    dim=-1
                ) if (neg_neural_vector is not None) and (pos_neural_vector is not None) else torch.concat(
                    [pos_symbolic_vector, neg_symbolic_vector],
                    dim=-1
                )
            )[:, 0]
        else:
            self.prob_score_model = lambda concept_latent_space, pos_symbolic_vector, neg_symbolic_vector, pos_neural_vector, neg_neural_vector, idx: self.sig(
                self.contrastive_scale[idx] * self._distance_metric(
                    neg_anchor=neg_symbolic_vector,
                    pos_anchor=pos_symbolic_vector,
                    latent=concept_latent_space,
                )
            )



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
        if self.learnable_orthogonal_dir:
            for i in range(self.n_concepts):
                for mode in [0, 1]:
                    loss += self.learnable_orthogonal_dir * torch.pow(
                        self._cos_similarity(self.concept_embeddings[i, mode, :], self.orthogonal_embeddings[i, mode, :]),
                        2,
                    ) / (self.n_concepts * 2)
        if 'per_class_mixing' in self.bottleneck_pooling and self.single_residual_vector and self.residual_weight_l2:
            loss += torch.norm(
                self.downstream_label_weights[:, self.n_concepts:]
            ) / (
                self.n_tasks * (
                    self.downstream_label_weights.shape[-1] - self.n_concepts
                )
            )

        if self.mix_residuals and (self._residual is not None) and self.residual_sep_loss:
            out_c_preds = []
            for i in range(self.n_concepts):
                input_to_scoring = self._residual[:, i, :, :].view(
                    self._residual.shape[0],
                    2 * self.extra_capacity,
                )
                out_c_preds.append(self.scoring_model(input_to_scoring))
            self._residual = None
            out_c_preds = torch.concat(out_c_preds, dim=-1)
            loss += self.residual_sep_loss * self.loss_concept(out_c_preds, c)

        self._residual = None
        return loss

    def _compute_task_loss(
        self,
        y,
        probs=None,
        pos_embeddings=None,
        neg_embeddings=None,
        y_pred_logits=None,
        **task_loss_kwargs,
    ):
        if y_pred_logits is not None:
            return self.task_loss_weight * self.loss_task(
            (
                y_pred_logits if y_pred_logits.shape[-1] > 1
                else y_pred_logits.reshape(-1)
            ),
            y,
        )
        if self.warmup_mode:
            y_logits = self.bypass_label_predictor(task_loss_kwargs['latent_space'])
        else:
            bottleneck = self._construct_c2y_input(
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                probs=probs,
                **task_loss_kwargs,
            )
            y_logits = self._predict_labels(bottleneck)
        return self.task_loss_weight * self.loss_task(
            (
                y_logits
                if y_logits.shape[-1] > 1 else
                y_logits.reshape(-1)
            ),
            y,
        )

    def on_before_backward(self, loss):
        if self.adversary_loss_weight == 0:
            # The nothing we need to do
            self._discr_loss = None
            return
        self.num_steps += 1

        self._discr_loss.backward(retain_graph=True)
        if (self.num_steps.detach() >= self.warmup_period):
            discriminator_params = {
                name for name, _ in self.discriminator.named_parameters()
            }
            self._predictor_grads_wrt_discr_loss = {
                name: param.grad.clone()
                for name, param in self.named_parameters()
                if name not in discriminator_params and param.grad is not None
            }
        else:
            self._predictor_grads_wrt_discr_loss = None
        self.discriminator_optimizer.step()
        self.discriminator_optimizer.zero_grad()
        self.optimizers().zero_grad()

    def on_before_optimizer_step(self, *args, **kwargs):
        if self.adversary_loss_weight == 0 or (self.num_steps.detach() < self.warmup_period):
            # The nothing we need to do
            return
        if self.adversary_loss_weight == "dyn":
            scale = torch.sqrt(self.num_steps.detach())
        else:
            scale = self.adversary_loss_weight
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name not in self._predictor_grads_wrt_discr_loss:
                    continue
                unit_protect = self._predictor_grads_wrt_discr_loss[name] / (torch.linalg.norm(self._predictor_grads_wrt_discr_loss[name]) + 1e-8)
                param.grad -= ((param.grad * unit_protect) * unit_protect).sum()
                param.grad -= scale * self._predictor_grads_wrt_discr_loss[name]
                if self.adversary_loss_weight == "dyn":
                    # Then also scale the learning rate magnitude to avoid divergence
                    param.grad *= 1 /(max(self.num_steps.detach()/self.dyn_scaling, 1))
        self._discr_loss = None
        self._predictor_grads_wrt_discr_loss = None

    def _mix(self, pos_embs, neg_embs, probs):
        if len(probs.shape) != len(pos_embs.shape):
            probs = probs.unsqueeze(-1)

        return probs * pos_embs + (1 - probs) * neg_embs

    def _construct_c2y_input(
        self,
        pos_embeddings,
        neg_embeddings,
        probs,
        **task_loss_kwargs,
    ):
        pred_concepts = self._mix(pos_embeddings, neg_embeddings, probs)
        return self._make_bottleneck(
            pred_concepts=pred_concepts,
            latent_space=task_loss_kwargs['latent_space'],
            probs=probs,
            train=True,
        )

    def manual_residual_scale(self, res_input, res):
        if self.training:
            return self._manual_residual_scale

        # Only do the OOD detection during inference
        if (self.residual_ood_detection == 1) or (self._residual_in_bn is None):
            return self._manual_residual_scale

        threshold = scipy.stats.chi2.ppf(
            self.residual_ood_detection,
            self._residual_in_bn.num_features,
        )
        normalized_input = torch.nn.functional.batch_norm(
            res_input,
            self._residual_in_bn.running_mean,
            self._residual_in_bn.running_var,
        )
        maha_distance = torch.pow(normalized_input, 2).sum(-1)
        mask = self._manual_residual_scale * (maha_distance <= threshold)
        while len(mask.shape) < len(res.shape):
            mask = mask.unsqueeze(-1)
        return mask

    def _make_bottleneck(
        self,
        pred_concepts,
        latent_space,
        probs,
        train=False,
    ):
        if self.bottleneck_pooling == 'additive':
            bottleneck = torch.zeros(
                (pred_concepts.shape[0], self.emb_size)
            ).to(pred_concepts.device)
            for i in range(self.n_concepts):
                bottleneck += pred_concepts[:, i, :]
        elif self.bottleneck_pooling == 'mean':
            bottleneck = torch.mean(pred_concepts, axis=1)
        elif self.bottleneck_pooling == 'weighted_mean':
            bottleneck = torch.zeros(
                (pred_concepts.shape[0], self.emb_size)
            ).to(pred_concepts.device)
            norm_factor = torch.sum(self.concept_pool_weights)
            for i in range(self.n_concepts):
                bottleneck += (
                    self.concept_pool_weights[i] * pred_concepts[:, i, :]
                )
            bottleneck = bottleneck / norm_factor
        elif self.bottleneck_pooling == 'concat':
            bottleneck = torch.flatten(
                pred_concepts,
                start_dim=1,
            )
            if self.extra_capacity_residual is not None:
                extra_capacity = self.extra_capacity_residual(latent_space, probs)
                bottleneck = torch.concat(
                    [bottleneck, extra_capacity],
                    dim=-1,
                )
        elif 'per_class_mixing' in self.bottleneck_pooling:
            bottleneck = pred_concepts
        else:
            raise ValueError(
                f'Unsupported bottleneck pooling "{self.bottleneck_pooling}".'
            )

        if self.single_residual_vector:
            if not self.conditional_residual:
                input_to_residual = latent_space
            else:
                mixed_emb = torch.sum(
                    self.mixture_coefficients.unsqueeze(0).unsqueeze(-1) * pred_concepts,
                    dim=1,
                )
                input_to_residual = torch.concat([latent_space, mixed_emb], dim=-1)
            if self._residual is None:
                res = self.residual_model(input_to_residual)
                self._residual = self.manual_residual_scale(input_to_residual, res) * res
            residual = self._residual
            if self.mix_residuals:
                probs = probs.unsqueeze(-1)
                mixed_embs = residual[:, :, 0, :] * probs + (1 - probs) * residual[:, :, 1, :]
                residual = torch.sum(
                    self._residual_mix_coefficients.unsqueeze(0).unsqueeze(-1) * mixed_embs,
                    dim=1,
                )
            if not self.adversary_loss_weight:
                # Then there is the chance we will drop this guy entirely!
                residual = self.residual_masking(torch.ones([1]*len(residual.shape)).to(residual)) * residual
            if len(bottleneck.shape) == 3:
                residual = residual.unsqueeze(1)
            bottleneck = torch.concat([bottleneck, residual], dim=1)
        return bottleneck


    def _distance_metric(self, neg_anchor, pos_anchor, latent):

        # if self.use_cosine_similarity:
        #     pos_projection = pos_anchor * dot(pos_anchor, latent)/(torch.norm(pos_anchor, p=2) + 1e-8)
        #     neg_projection = neg_anchor * dot(neg_anchor, latent)/(torch.norm(neg_anchor, p=2) + 1e-8)
        #     neg_dist = (neg_anchor - neg_projection).pow(2).sum(-1).sqrt()
        #     pos_dist = (pos_anchor - pos_projection).pow(2).sum(-1).sqrt()
        #     return neg_dist - pos_dist
        # else:
        neg_dist = (neg_anchor - latent).pow(2).sum(-1).sqrt()
        pos_dist = (pos_anchor - latent).pow(2).sum(-1).sqrt()
        return neg_dist - pos_dist


    def _orthogonal_projection(
        self,
        latent,
        base,
        concept_idx,
        positive,
    ):
        if self.learnable_orthogonal_dir:
            orth_vector = self.orthogonal_embeddings[concept_idx, 0 if positive else 1, :].expand(latent.shape[0], -1)
            orth_scale = self.orthogonal_scale_models[concept_idx](latent)
            return orth_vector * orth_scale

        y_y = dot(base, base)
        x_y = dot(latent, base)
        orth_proj = latent - (x_y * base) / (y_y + 1e-8)
        return orth_proj

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if latent is None:
            pre_c = self.pre_concept_model(x)
            model_latent_space = pre_c
            c_sem = []
            pred_concepts = []
            pos_embs = []
            neg_embs = []
            if (self.residual_model is not None):
                res = self.residual_model(pre_c)
                self._residual = self.manual_residual_scale(pre_c, res) * res

            # First predict all the concept probabilities
            for i, concept_emb_generator in enumerate(
                self.concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                concept_latent_space = concept_emb_generator(pre_c)
                if self.shared_emb_generator:
                    model_latent_space = concept_latent_space
                # [Shape: (B, emb_size)]
                if self.use_triplet_loss:
                    pos_symbolic_vector = torch.unsqueeze(
                        self.concept_embeddings[i, 0, :],
                        dim=0,
                    ).expand(concept_latent_space.shape[0], -1)
                else:
                    pos_symbolic_vector = torch.unsqueeze(
                        self.concept_embeddings[i, :],
                        dim=0,
                    ).expand(concept_latent_space.shape[0], -1)

                if self._residual is not None:
                    if len(self._residual.shape) == 4:
                        pos_neural_vector = self._residual[:, i, 0, :]
                    elif len(self._residual.shape) == 3:
                        pos_neural_vector = self._residual[:, i, :]
                    elif len(self._residual.shape) == 2:
                        pos_neural_vector = self._residual
                    else:
                        pos_neural_vector = None
                elif self.single_residual_vector:
                    pos_neural_vector = None
                else:
                    pos_neural_vector = self._orthogonal_projection(
                        latent=concept_latent_space,
                        base=pos_symbolic_vector,
                        concept_idx=i,
                        positive=True,
                    )
                if (pos_neural_vector is None) or (self.mix_residuals):
                    current_pos_embs = pos_symbolic_vector
                else:
                    current_pos_embs = torch.concat(
                        [
                            pos_symbolic_vector,
                            pos_neural_vector,

                        ],
                        dim=-1,
                    )

                if self.use_triplet_loss:
                    neg_symbolic_vector = torch.unsqueeze(
                        self.concept_embeddings[i, 1, :],
                        dim=0,
                    ).expand(concept_latent_space.shape[0], -1)
                else:
                    neg_symbolic_vector = -pos_symbolic_vector

                if self._residual is not None:
                    if len(self._residual.shape) == 4:
                        neg_neural_vector = self._residual[:, i, 1, :]
                    elif len(self._residual.shape) == 3:
                        neg_neural_vector = self._residual[:, i, :]
                    elif len(self._residual.shape) == 2:
                        neg_neural_vector = self._residual
                    else:
                        neg_neural_vector = None
                elif self.single_residual_vector:
                    neg_neural_vector = None
                else:
                    neg_neural_vector = self._orthogonal_projection(
                        latent=concept_latent_space,
                        base=neg_symbolic_vector,
                        concept_idx=i,
                        positive=False,
                    )
                if (neg_neural_vector is None) or self.mix_residuals:
                    current_neg_embs = neg_symbolic_vector
                else:
                    current_neg_embs = torch.concat(
                        [
                            neg_symbolic_vector,
                            neg_neural_vector,

                        ],
                        dim=-1,
                    )

                prob = self.prob_score_model(
                    concept_latent_space,
                    pos_symbolic_vector,
                    neg_symbolic_vector,
                    pos_neural_vector,
                    neg_neural_vector,
                    i,
                )

                # [Shape: (B, 1)]
                if len(prob.shape) != 2:
                    prob = torch.unsqueeze(prob, dim=-1)


                mixed_embs = self._mix(current_pos_embs, current_neg_embs, prob)

                pos_embs.append(current_pos_embs.unsqueeze(1))
                neg_embs.append(current_neg_embs.unsqueeze(1))

                c_sem.append(prob)
                pred_concepts.append(
                    torch.unsqueeze(mixed_embs, dim=1)
                )

            c_sem = torch.cat(c_sem, dim=-1)
            pos_embs = torch.cat(pos_embs, dim=1)
            neg_embs = torch.cat(neg_embs, dim=1)
            pred_concepts = torch.cat(pred_concepts, dim=1)
            latent = pre_c,  c_sem, pred_concepts, model_latent_space, pos_embs, neg_embs
        else:
            pre_c, c_sem, pred_concepts, model_latent_space, pos_embs, neg_embs = latent
        self._task_loss_kwargs.update({
            'latent_space': model_latent_space,
        })
        return c_sem, pos_embs, neg_embs, {
            "pred_concepts": pred_concepts,
            "pre_c": pre_c,
            "latent_space": model_latent_space,
            "latent": latent,
        }

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        self._residual = None
        if self.simplified_mode:
            return self._simplified_run_step(
                batch=batch,
                batch_idx=batch_idx,
                train=train,
                intervention_idxs=intervention_idxs,
            )

        x, y, (c, g, competencies, prev_interventions) = self._unpack_batch(
            batch
        )
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
            output_interventions=True,
        )
        c_sem, bottleneck, y_logits = outputs[0], outputs[1], outputs[2]
        # prev_interventions will contain the RandInt intervention mask if
        # we are running this a train time!
        prev_interventions = outputs[3]
        latent = outputs[4]
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]

        # Then the rollout and imitation learning losses
        c_used = c

        if train and (intervention_idxs is None) and (not getattr(
            self,
            'warmup_mode',
            False,
        )):
            (
                intervention_loss,
                intervention_task_loss,
                int_mask_accuracy,
            ) = self._intervention_rollout_loss(
                c=c_used,
                c_pred=c_sem,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                y=y,
                y_pred_logits=y_logits,
                prev_interventions=prev_interventions,
                competencies=competencies,
                **self._task_loss_kwargs,
            )
        else:
            intervention_loss = 0
            intervention_task_loss = self._compute_task_loss(
                y=y,
                probs=c_sem,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                **self._task_loss_kwargs,
            )
            int_mask_accuracy = 0


        if isinstance(intervention_task_loss, (float, int)):
            intervention_task_loss_scalar = (
                self.intervention_task_loss_weight * intervention_task_loss
            )
        else:
            intervention_task_loss_scalar = (
                self.intervention_task_loss_weight *
                intervention_task_loss.detach()
            )

        if isinstance(intervention_loss, (float, int)):
            intervention_loss_scalar = \
                self.intervention_weight * intervention_loss
        else:
            intervention_loss_scalar = \
                self.intervention_weight * intervention_loss.detach()


        # Finally, compute the concept loss
        concept_loss = self._compute_concept_loss(
            c=c,
            c_pred=c_sem,
        )
        if isinstance(concept_loss, (float, int)):
            concept_loss_scalar = self.concept_loss_weight * concept_loss
        else:
            concept_loss_scalar = \
                self.concept_loss_weight * concept_loss.detach()

        # Adversarial loss stuff!!
        extra_results = {}
        if self.adversary_loss_weight:
            if len(bottleneck.shape) == 2:
                residual = bottleneck[:, -self.extra_capacity:]
            else:
                residual = bottleneck[:, self.n_concepts, :]
            adversarial_preds = self.discriminator(residual)
            self._discr_loss = self.loss_concept(adversarial_preds, c)
            (adv_c_accuracy, adv_c_auc, adv_c_f1), _ = compute_accuracy(
                c_pred=adversarial_preds,
                y_pred=None,
                c_true=c,
                y_true=y,
            )
            extra_results['adv_loss'] = self._discr_loss.detach()
            extra_results['adv_c_auc'] = adv_c_auc

        loss = (
            self.concept_loss_weight * concept_loss +
            self.intervention_weight * intervention_loss +
            self.intervention_task_loss_weight * intervention_task_loss
        )

        loss += self._extra_losses(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            c_pred=bottleneck,
            y_pred=y_logits,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "mask_accuracy": int_mask_accuracy,
            "concept_loss": concept_loss_scalar,
            "intervention_task_loss": intervention_task_loss_scalar,
            "task_loss": 0, # As the actual task loss is included above!
            "intervention_loss": intervention_loss_scalar,
            "loss": loss.detach() if not isinstance(loss, float) else loss,
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
            "horizon_limit": self.horizon_limit.detach().cpu().numpy()[0],
        }
        result["current_steps"] = \
            self.current_steps.detach().cpu().numpy()[0]
        result.update(extra_results)
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            for top_k_val in self.top_k_accuracy:
                y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                    y_true,
                    y_pred,
                    k=top_k_val,
                    labels=labels,
                )
                result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result

    def _loss_mean(self, losses, y, loss_weights=None):
        if loss_weights is None:
            loss_weights = torch.ones_like(y)
        if self.loss_task.weight is not None:
            norm_constant = torch.gather(self.loss_task.weight, 0, y)
            return torch.sum(losses, dim=-1) / torch.sum(
                loss_weights * norm_constant,
                dim=-1,
            )
        return torch.sum(losses, dim=-1) / torch.sum(loss_weights, dim=-1)

    def _simplified_run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        x, y, (c, g, competencies, prev_interventions) = self._unpack_batch(
            batch
        )

        int_probs = []
        og_int_probs = self.training_intervention_prob
        if train and self.warmup_mode:
            int_probs = [0]
        elif isinstance(self.training_intervention_prob, list) and train:
            int_probs = self.training_intervention_prob
        elif train:
            int_probs = [self.training_intervention_prob]
        else:
            int_probs = [0]

        task_loss = 0.0
        loss_weights = 0.0
        c_sem = None
        c_logits = None
        for int_prob in int_probs:
            self.training_intervention_prob = int_prob
            outputs = self._forward(
                x,
                intervention_idxs=intervention_idxs,
                c=c,
                y=y,
                train=train,
                competencies=competencies,
                prev_interventions=prev_interventions,
                output_interventions=True,
            )
            if c_sem is None:
                c_sem, c_logits = outputs[0], outputs[1]

            y_logits = outputs[2]
            int_mask = outputs[3]
            current_task_loss = self._compute_task_loss(
                y=y,
                y_pred_logits=y_logits,
                **self._task_loss_kwargs,
            )
            if int_mask is not None:
                scaling = torch.pow(
                    self.intervention_task_discount,
                    torch.sum(int_mask, dim=-1),
                )
                loss_weights += scaling
                task_loss += current_task_loss * scaling
            else:
                task_loss += current_task_loss
                loss_weights += torch.ones_like(current_task_loss)
        task_loss = self._loss_mean(
            task_loss,
            y=y,
            loss_weights=loss_weights,
        )
        if not isinstance(task_loss, (float, int)):
            task_loss_scalar = task_loss.detach()
        else:
            task_loss_scalar = task_loss
        self.training_intervention_prob = og_int_probs

        if self.concept_loss_weight != 0:
            # We separate this so that we are allowed to
            # use arbitrary activations (i.e., not necessarily in [0, 1])
            # whenever no concept supervision is provided
            # Will only compute the concept loss for concepts whose certainty
            # values are fully given
            concept_loss = self.loss_concept(c_sem, c)
            concept_loss_scalar = concept_loss.detach()
            loss = self.concept_loss_weight * concept_loss + task_loss + \
                self._extra_losses(
                    x=x,
                    y=y,
                    c=c,
                    c_sem=c_sem,
                    c_pred=c_logits,
                    y_pred=y_logits,
                    competencies=competencies,
                    prev_interventions=prev_interventions,
                )
        else:
            loss = task_loss + self._extra_losses(
                x=x,
                y=y,
                c=c,
                c_sem=c_sem,
                c_pred=c_logits,
                y_pred=y_logits,
                competencies=competencies,
                prev_interventions=prev_interventions,
            )
            concept_loss_scalar = 0.0
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            if isinstance(self.top_k_accuracy, int):
                top_k_accuracy = list(range(1, self.top_k_accuracy))
            else:
                top_k_accuracy = self.top_k_accuracy

            for top_k_val in top_k_accuracy:
                if top_k_val:
                    y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                        y_true,
                        y_pred,
                        k=top_k_val,
                        labels=labels,
                    )
                    result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result

    def _predict_labels(self, bottleneck):
        if 'per_class_mixing' in self.bottleneck_pooling:
            # Shape (B, k, m)
            concept_vectors = torch.nn.functional.leaky_relu(bottleneck[:, :self.n_concepts, :])
            if self.single_residual_vector:
                # Shape (B, n_residuals, m)
                residual = torch.nn.functional.leaky_relu(
                    bottleneck[:, self.n_concepts:, :]
                )
                # Shape (B, L, n_residuals, m)
                residual = residual.unsqueeze(1).expand(-1, self.n_tasks, -1, -1)
            else:
                residual = 0

            # self.downstream_label_weights has shape (L, k)
            # We want shape (B, L, m)
            mixed_emb = torch.einsum('bkm,lk->blm', concept_vectors, self.downstream_label_weights[:, :self.n_concepts])
            if self.single_residual_vector:
                mixed_emb += (
                    residual * self.downstream_label_weights[:, self.n_concepts:].unsqueeze(0).unsqueeze(-1)
                ).sum(dim=2)


            if self.bottleneck_pooling == 'per_class_mixing_shared':
                # shape (1, emb_size)
                raise ValueError('Unsupported for now!!')
                class_embs = self.class_embeddings[task_idx:task_idx+1, :].expand(
                    mixed_emb.shape[0],
                    -1,
                )
                # shape (B, 2 * emb_size)
                mixed_emb = torch.concat(
                    [mixed_emb, class_embs],
                    dim=-1,
                )
                logits.append(self.c2y_model(mixed_emb))
            else:
                # Shape: (B, L)
                y_pred = self.c2y_model(mixed_emb).squeeze(-1)

            # logits = []
            # for task_idx in range(self.n_tasks):
            #     # shape (B, emb_size)
            #     if self.single_residual_vector:
            #         residual = torch.nn.functional.leaky_relu(bottleneck[:, self.n_concepts, :])
            #     else:
            #         residual = 0
            #     concept_vectors = torch.nn.functional.leaky_relu(bottleneck[:, :self.n_concepts, :])
            #     mixed_emb = torch.sum(
            #         self.downstream_label_weights.unsqueeze(0)[:, task_idx:task_idx+1, :self.n_concepts].transpose(1, 2) * concept_vectors,
            #         dim=1,
            #     )
            #     if self.single_residual_vector:
            #         mixed_emb += self.downstream_label_weights[task_idx, self.n_concepts] * residual
            #     if self.bottleneck_pooling == 'per_class_mixing_shared':
            #         # shape (1, emb_size)
            #         class_embs = self.class_embeddings[task_idx:task_idx+1, :].expand(
            #             mixed_emb.shape[0],
            #             -1,
            #         )
            #         # shape (B, 2 * emb_size)
            #         mixed_emb = torch.concat(
            #             [mixed_emb, class_embs],
            #             dim=-1,
            #         )
            #         logits.append(self.c2y_model(mixed_emb))
            #     else:
            #         logits.append(self.c2y_model[task_idx](mixed_emb))

            # if len(logits) > 1:
            #     y_pred = torch.concat(logits, dim=-1)
            # else:
            #     y_pred = logits[0]

        else:
            y_pred = self.c2y_model(bottleneck)
        return y_pred

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
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )

        c_sem, pos_embs, neg_embs, out_kwargs = self._generate_concept_embeddings(
            x=x,
            latent=latent,
            training=train,
        )

        # Now include any interventions that we may want to perform!
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=pos_embs,
                neg_embeddings=neg_embs,
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
                **self._task_loss_kwargs,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )

        else:
            c_int = c
        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
            )

        # Then, time to do the mixing between the positive and the
        # negative embeddings
        latent_space = out_kwargs.pop('latent_space')
        _, intervention_idxs, bottleneck = \
            self._after_interventions(
                c_sem,
                pos_embeddings=pos_embs,
                neg_embeddings=neg_embs,
                latent_space=latent_space,
                intervention_idxs=intervention_idxs,
                c_true=c_int,
                train=train,
                competencies=competencies,
                **out_kwargs
            )

        y_pred = self._predict_labels(bottleneck)

        tail_results = []
        if output_interventions:
            if (
                (intervention_idxs is not None) and
                isinstance(intervention_idxs, np.ndarray)
            ):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            if "latent" in out_kwargs:
                latent = out_kwargs['latent']
            tail_results.append(latent)
        if output_embeddings and (not pos_embs is None) and (
            not neg_embs is None
        ):
            tail_results.append(pos_embs)
            tail_results.append(neg_embs)

        return tuple([c_sem, bottleneck, y_pred] + tail_results)



    def freeze_concept_embeddings(self):
        self.concept_embeddings.requires_grad = False

    def unfreeze_concept_embeddings(self):
        self.concept_embeddings.requires_grad = True

    def freeze_residual(self):
        pass
        # for param in self.residual_model.parameters():
        #     param.requires_grad = False
        # if self.conditional_residual:
        #     self.mixture_coefficients.requires_grad = False

    def unfreeze_residual(self):
        pass
        # for param in self.residual_model.parameters():
        #     param.requires_grad = True
        # if self.conditional_residual:
        #     self.mixture_coefficients.requires_grad = True

    def freeze_label_predictor(self):
        if 'per_class_mixing' in self.bottleneck_pooling:
            for submodel in self.c2y_model:
                for param in submodel.parameters():
                    param.requires_grad = False
            self.downstream_label_weights.requires_grad = False
            if self.bottleneck_pooling == 'per_class_mixing_shared':
                self.class_embeddings.requires_grad = False
        else:
            for param in self.c2y_model.parameters():
                param.requires_grad = False

    def unfreeze_label_predictor(self):
        if 'per_class_mixing' in self.bottleneck_pooling:
            for submodel in self.c2y_model:
                for param in submodel.parameters():
                    param.requires_grad = True
            self.downstream_label_weights.requires_grad = True
            if self.bottleneck_pooling == 'per_class_mixing_shared':
                self.class_embeddings.requires_grad = True
        else:
            for param in self.c2y_model.parameters():
                param.requires_grad = True

    def freeze_backbone(self, freeze_emb_generators=False):
        for param in self.pre_concept_model.parameters():
            param.requires_grad = False
        if freeze_emb_generators:
            # And the concept generators
            for emb_generator in self.concept_emb_generators:
                for param in emb_generator.parameters():
                    param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.pre_concept_model.parameters():
            param.requires_grad = False
        # And the concept generators
        for emb_generator in self.concept_emb_generators:
            for param in emb_generator.parameters():
                param.requires_grad = False