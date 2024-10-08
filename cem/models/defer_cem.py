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


class DeferConceptEmbeddingModel(IntAwareConceptEmbeddingModel):
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

        top_k_accuracy=None,


        # New stuff
        mixcem_c2y_scaling=1,
        dynamic_c2y_scaling=1,

        # Mixing
        bottleneck_pooling='concat',
        fixed_embeddings=False,
        initial_concept_embeddings=None,
        fixed_scale=None,

        simplified_mode=False,

        shared_emb_generator=True,
        dynamic_ood_detection=1,
        dynamic_weights_reg=0,
        dynamic_weights_reg_norm=2,
        dynamic_activations_reg=0,
        dynamic_activations_reg_norm=2,
        dynamic_drop_prob=0,
        conditional_pred_mixture=False,
        dyn_from_shared_space=True,
        mode='joint',
        probability_mode='joint',
        mix_before_predictor=False,
        joint_model_pooling=None,
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        if bottleneck_pooling == 'pcm':
            bottleneck_pooling = 'per_class_mixing'
        if joint_model_pooling == 'pcm':
            joint_model_pooling = 'per_class_mixing'
        self.bottleneck_pooling = bottleneck_pooling
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
            self.bottleneck_size = emb_size
        else:
            self.bottleneck_size = emb_size * n_concepts

        # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # AFTER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.dynamic_model = IntAwareConceptEmbeddingModel(
        #     n_concepts=n_concepts,
        #     n_tasks=n_tasks,
        #     emb_size=emb_size,
        #     training_intervention_prob=training_intervention_prob,
        #     embedding_activation=embedding_activation,
        #     concept_loss_weight=concept_loss_weight,
        #     c2y_model=c2y_model,
        #     c2y_layers=c2y_layers,
        #     c_extractor_arch=c_extractor_arch,
        #     output_latent=output_latent,
        #     optimizer=optimizer,
        #     momentum=momentum,
        #     learning_rate=learning_rate,
        #     weight_decay=weight_decay,
        #     lr_scheduler_factor=lr_scheduler_factor,
        #     lr_scheduler_patience=lr_scheduler_patience,
        #     weight_loss=weight_loss,
        #     task_class_weights=task_class_weights,
        #     active_intervention_values=active_intervention_values,
        #     inactive_intervention_values=inactive_intervention_values,
        #     intervention_policy=intervention_policy,
        #     output_interventions=output_interventions,
        #     top_k_accuracy=top_k_accuracy,
        #     intervention_task_discount=intervention_task_discount,
        #     intervention_weight=intervention_weight,
        #     concept_map=concept_map,
        #     use_concept_groups=use_concept_groups,
        #     rollout_init_steps=rollout_init_steps,
        #     int_model_layers=int_model_layers,
        #     int_model_use_bn=int_model_use_bn,
        #     num_rollouts=num_rollouts,
        #     max_horizon=max_horizon,
        #     initial_horizon=initial_horizon,
        #     horizon_rate=horizon_rate,
        #     intervention_discount=intervention_discount,
        #     include_only_last_trajectory_loss=include_only_last_trajectory_loss,
        #     task_loss_weight=task_loss_weight,
        #     intervention_task_loss_weight=intervention_task_loss_weight,
        # )
        # END OF CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if c_extractor_arch == "identity":
            self.pre_concept_model = lambda x: x
        else:
            self.pre_concept_model = c_extractor_arch(output_dim=None)
        if self.pre_concept_model == "identity":
            c_extractor_arch = "identity"
            self.pre_concept_model = lambda x: x
        # AFTER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.pre_concept_model = self.dynamic_model.pre_concept_model
        # END OF CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



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
        self.emb_size = emb_size
        self.concept_embeddings = torch.nn.Parameter(
            initial_concept_embeddings,
            requires_grad=(not fixed_embeddings),
        )

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
        # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.dynamic_emb_generators = torch.nn.ModuleList()
        # AFTER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.dynamic_emb_generators = self.dynamic_model.concept_context_generators
        # END OF CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

            # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.dyn_from_shared_space = dyn_from_shared_space
            self.dynamic_emb_generators.append(
                torch.nn.Sequential(*[
                    # torch.nn.BatchNorm1d(num_features=emb_size),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(
                        emb_size if self.dyn_from_shared_space else n_in_feats,
                        2*emb_size,
                    ),
                    emb_act,
                    torch.nn.Unflatten(-1, (2, emb_size))
                ])
            )
            # END OF CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.shared_emb_generator:
                if len(self.concept_emb_generators) == 0:
                    self.concept_emb_generators.append(
                        torch.nn.Sequential(*[
                            # torch.nn.BatchNorm1d(num_features=n_in_feats),
                            # torch.nn.LeakyReLU(),
                            torch.nn.Linear(
                                n_in_feats,
                                emb_size,
                            ),
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
                        # torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            n_in_feats,
                            emb_size,
                        ),
                        emb_act,
                    ])
                )

        self.dynamic_masking = lambda x: x
        if dynamic_drop_prob:
            if isinstance(dynamic_drop_prob, (int, float)):
                self.dynamic_masking = DynamicDropoutLayer(p=dynamic_drop_prob, rate=1)
            elif isinstance(dynamic_drop_prob, str) and dynamic_drop_prob.startswith('dyn_'):
                rate, min_prob = dynamic_drop_prob.split("_")[1:]
                self.dynamic_masking = DynamicDropoutLayer(
                    p=1,
                    rate=float(rate),
                    min_prob=float(min_prob),
                )
            else:
                raise ValueError(f'Unsupported dynamic_drop_prob {dynamic_drop_prob}')

        # Make prediction scaling model
        self.conditional_pred_mixture = conditional_pred_mixture
        mixture_input_size = self.emb_size
        if conditional_pred_mixture == 'both':
            mixture_input_size += 2*self.bottleneck_size
        elif conditional_pred_mixture in ['mixcem', 'dynamic']:
            mixture_input_size += self.bottleneck_size
        elif conditional_pred_mixture == 'none':
            pass
        else:
            raise ValueError(
                f'Invalid conditional_pred_mixture {conditional_pred_mixture}'
            )
        units = [mixture_input_size] + (c2y_layers or []) + [
            n_concepts if mix_before_predictor else 1
        ]
        layers = [torch.nn.BatchNorm1d(num_features=mixture_input_size)]
        for i in range(1, len(units)):
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != (len(units) - 1):
                layers.append(torch.nn.LeakyReLU())
        self.pred_mixture_model = torch.nn.Sequential(*layers)
        self._coeffs = None
        self.dynamic_weights_reg = dynamic_weights_reg
        self.dynamic_weights_reg_norm = dynamic_weights_reg_norm
        self.dynamic_activations_reg = dynamic_activations_reg
        self.dynamic_activations_reg_norm = dynamic_activations_reg_norm

        # Make predictor model
        self._mixcem_c2y_scaling = mixcem_c2y_scaling
        self._dynamic_c2y_scaling = dynamic_c2y_scaling
        if 'per_class_mixing' in self.bottleneck_pooling:
            self.mixcem_downstream_label_weights = torch.nn.Parameter(
                torch.rand((n_tasks, self.n_concepts)),
                requires_grad=True,
            )
            self.dynamic_downstream_label_weights = torch.nn.Parameter(
                torch.rand((n_tasks, self.n_concepts)),
                requires_grad=True,
            )
            pred_out_size = 1
            if self.bottleneck_pooling == 'per_class_mixing_shared':
                self.mixcem_class_embeddings = torch.nn.Parameter(
                    torch.rand((n_tasks, 2*self.emb_size)),
                    requires_grad=True,
                )
                self.dynamic_class_embeddings = torch.nn.Parameter(
                    torch.rand((n_tasks, 2*self.emb_size)),
                    requires_grad=True,
                )
                pred_input_size = self.emb_size * 2
            else:
                pred_input_size = self.emb_size
        else:
            pred_input_size = self.bottleneck_size
            pred_out_size = n_tasks
        units = [pred_input_size] + (c2y_layers or []) + [pred_out_size]
        layers = []
        for i in range(1, len(units)):
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != (len(units) - 1):
                layers.append(torch.nn.LeakyReLU())
        self.mixcem_c2y_model = torch.nn.Sequential(*layers)
        # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        layers = []
        for i in range(1, len(units)):
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != (len(units) - 1):
                layers.append(torch.nn.LeakyReLU())
        self.dynamic_c2y_model = torch.nn.Sequential(*layers)
        # AFTER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.dynamic_c2y_model = self.dynamic_model.c2y_model
        # END OF CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
        self.dynamic_ood_detection = dynamic_ood_detection

        self.mode = mode
        self.probability_mode = probability_mode
        self.simplified_mode = simplified_mode

        # Built-in CEM stuff
        # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self._dynamic_prob_score_models = torch.nn.ModuleList([
                torch.nn.Sequential(
                torch.nn.Linear(2*(emb_size), 1),
                torch.nn.Sigmoid(),
            ) for _ in range(n_concepts)
        ])
        # AFTER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self._dynamic_prob_score_models = self.dynamic_model.concept_prob_generators
        # END OF CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Probability functions
        # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.dynamic_prob_score_model = lambda concept_latent_space, mixcem_pos_emb, mixcem_neg_embedding, dynamic_pos_embedding, dynamic_neg_embedding, idx: self._dynamic_prob_score_models[idx](
            torch.concat([dynamic_pos_embedding, dynamic_neg_embedding], dim=-1)
        )
        # AFTER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.dynamic_prob_score_model = lambda concept_latent_space, mixcem_pos_emb, mixcem_neg_embedding, dynamic_pos_embedding, dynamic_neg_embedding, idx: self.sig(self._dynamic_prob_score_models[idx](
        #     torch.concat([dynamic_pos_embedding, dynamic_neg_embedding], dim=-1)
        # ))
        # END OF CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.mixcem_prob_score_model = lambda concept_latent_space, mixcem_pos_emb, mixcem_neg_embedding, dynamic_pos_embedding, dynamic_neg_embedding, idx: self.sig(
            self.contrastive_scale[idx] * self._distance_metric(
                neg_anchor=mixcem_neg_embedding,
                pos_anchor=mixcem_pos_emb,
                latent=concept_latent_space,
            )
        )


        self.mix_before_predictor = mix_before_predictor
        self.joint_model_pooling = joint_model_pooling or bottleneck_pooling
        if mix_before_predictor:
            if joint_model_pooling == bottleneck_pooling:
               self.joint_c2y_model = self.mixcem_c2y_model
            else:
                if 'per_class_mixing' in joint_model_pooling:
                    self.joint_downstream_label_weights = torch.nn.Parameter(
                        torch.rand((n_tasks, self.n_concepts)),
                        requires_grad=True,
                    )
                    pred_out_size = 1
                    if joint_model_pooling == 'per_class_mixing_shared':
                        self.joint_class_embeddings = torch.nn.Parameter(
                            torch.rand((n_tasks, 2*self.emb_size)),
                            requires_grad=True,
                        )
                        pred_input_size = self.emb_size * 2
                    else:
                        pred_input_size = self.emb_size
                else:
                    pred_input_size = self.bottleneck_size
                    pred_out_size = n_tasks
                units = [pred_input_size] + (c2y_layers or []) + [pred_out_size]
                layers = []
                for i in range(1, len(units)):
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                    if i != (len(units) - 1):
                        layers.append(torch.nn.LeakyReLU())
                self.joint_c2y_model = torch.nn.Sequential(*layers)

        # INTCEM STUFF!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.num_rollouts = num_rollouts
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map

        # Else we construct it here directly
        units = [
            self.bottleneck_size +
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

        self.mixcem_concept_rank_model = torch.nn.Sequential(*layers)

        # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        layers = [torch.nn.Flatten(start_dim=1, end_dim=-1)]
        for i in range(1, len(units)):
            if int_model_use_bn:
                layers.append(
                    torch.nn.BatchNorm1d(num_features=units[i-1]),
                )
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())
        self.dynamic_concept_rank_model = torch.nn.Sequential(*layers)
        # AFTER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.dynamic_concept_rank_model = self.dynamic_model.concept_rank_model
        # END OF CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        units = [
            2*self.bottleneck_size +
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

        self.merged_concept_rank_model = torch.nn.Sequential(*layers)

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


    def concept_rank_model(self, x):
        if self.mode == 'joint':
            return self.merged_concept_rank_model(x)

        # Else the input has both the mixcem and the dynamic embeddings
        # flattened in it (on top of the prev interventions!)
        prev_interventions = x[:, -self.n_concepts:]
        flattened_embeddings = x[:, :-self.n_concepts]

        if self.mode == 'mixcem':
            used_embeddings = flattened_embeddings[:, :flattened_embeddings.shape[-1]//2].view(
                flattened_embeddings.shape[0],
                self.n_concepts,
                self.emb_size,
            ).view(prev_interventions.shape[0], -1)
            used_model = self.mixcem_concept_rank_model
        else:
            used_embeddings = flattened_embeddings[:, flattened_embeddings.shape[-1]//2:].view(
                flattened_embeddings.shape[0],
                self.n_concepts,
                self.emb_size,
            ).view(prev_interventions.shape[0], -1)
            used_model = self.dynamic_concept_rank_model
        return used_model(
            torch.concat([used_embeddings, prev_interventions], dim=-1)
        )

    def c2y_model(self, mixcem_bottleneck, dynamic_bottleneck, latent_space):
        pred_mixture_inputs = latent_space
        if self.conditional_pred_mixture == 'both':
            pred_mixture_inputs = torch.concat(
                [
                    pred_mixture_inputs,
                    mixcem_bottleneck.view(mixcem_bottleneck.shape[0], -1),
                    dynamic_bottleneck.view(dynamic_bottleneck.shape[0], -1)
                ],
                dim=-1,
            )
        elif self.conditional_pred_mixture == 'mixcem':
            pred_mixture_inputs = torch.concat(
                [
                    pred_mixture_inputs,
                    mixcem_bottleneck.view(mixcem_bottleneck.shape[0], -1),
                ],
                dim=-1,
            )
        elif self.conditional_pred_mixture == 'dynamic':
            pred_mixture_inputs = torch.concat(
                [
                    pred_mixture_inputs,
                    dynamic_bottleneck.view(dynamic_bottleneck.shape[0], -1),
                ],
                dim=-1,
            )
        coeffs = self.pred_mixture_model(pred_mixture_inputs)
        self._coeffs = coeffs
        if self.mode == 'joint' and self.mix_before_predictor:
            # Then we mix at the embedding level with a single predictor
            if len(mixcem_bottleneck.shape) == 2:
                mixcem_vectors = mixcem_bottleneck.view(mixcem_bottleneck.shape[0], self.n_concepts, self.emb_size)
            else:
                mixcem_vectors = mixcem_bottleneck
            if len(dynamic_bottleneck.shape) == 2:
                dynamic_vectors = self.dynamic_masking(
                    dynamic_bottleneck
                ).view(dynamic_bottleneck.shape[0], self.n_concepts, self.emb_size)
            else:
                 dynamic_vectors = self.dynamic_masking(
                    dynamic_bottleneck
                )
            mixed_embeddings = self._mix(
                dynamic_vectors,
                self._mixcem_c2y_scaling * mixcem_vectors,
                coeffs.unsqueeze(-1),
            )
            if self.bottleneck_pooling != self.joint_model_pooling and (
                'per_class_mixing' in self.joint_model_pooling
            ):
                mixed_embeddings = torch.einsum(
                    'bkm,lk->blm',
                    mixed_embeddings,
                    self.joint_downstream_label_weights
                )

                if self.bottleneck_pooling == 'per_class_mixing_shared':
                    raise ValueError('Unsupported for now!!')
            preds = self.joint_c2y_model(
                mixed_embeddings
            )
            if len(preds.shape) == 3:
                preds = preds.squeeze(-1)
            return preds

        # Else we mix at the logits level
        mixcem_preds = self.mixcem_c2y_model(mixcem_bottleneck)
        if len(mixcem_preds.shape) == 3:
            mixcem_preds = mixcem_preds.squeeze(-1)
        dynamic_preds = self.dynamic_c2y_model(dynamic_bottleneck)
        if len(dynamic_preds.shape) == 3:
            dynamic_preds = dynamic_preds.squeeze(-1)
        if self.mode == 'mixcem':
            return mixcem_preds
        elif self.mode == 'dynamic':
            return dynamic_preds
        return  self._mixcem_c2y_scaling * (1 - coeffs) * mixcem_preds + (
            self._dynamic_c2y_scaling * coeffs * self.dynamic_masking(
                dynamic_preds
            )
        )

    def predict_probs(
        self,
        concept_latent_space,
        mixcem_pos_emb,
        mixcem_neg_embedding,
        dynamic_pos_embedding,
        dynamic_neg_embedding,
        idx,
    ):
        dynamic_c_probs = self.dynamic_prob_score_model(
            concept_latent_space=concept_latent_space,
            mixcem_pos_emb=mixcem_pos_emb,
            mixcem_neg_embedding=mixcem_neg_embedding,
            dynamic_pos_embedding=dynamic_pos_embedding,
            dynamic_neg_embedding=dynamic_neg_embedding,
            idx=idx,
        )
        mixcem_c_probs = self.mixcem_prob_score_model(
            concept_latent_space=concept_latent_space,
            mixcem_pos_emb=mixcem_pos_emb,
            mixcem_neg_embedding=mixcem_neg_embedding,
            dynamic_pos_embedding=dynamic_pos_embedding,
            dynamic_neg_embedding=dynamic_neg_embedding,
            idx=idx,
        ).unsqueeze(-1)
        dynamic_scaling = 1
        mixcem_scaling = 1
        if self.mode == 'mixcem':
            dynamic_scaling = 0
        elif self.mode == 'dynamic':
            mixcem_scaling = 0
        elif self.mode == 'joint':
            if self.probability_mode == 'joint':
                dynamic_scaling = 0.5
                mixcem_scaling = 0.5
            elif self.probability_mode == 'dynamic':
                dynamic_scaling = 1
                mixcem_scaling = 0
            elif self.probability_mode == 'mixcem':
                dynamic_scaling = 0
                mixcem_scaling = 1
        else:
            raise ValueError(f'Unsupported mode {self.mode}')
        return (
            dynamic_scaling * dynamic_c_probs +
            mixcem_scaling * mixcem_c_probs
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
        if self.mode != 'joint':
            return loss

        if self.dynamic_weights_reg:
            norm_factor = 0
            for _, params in self.dynamic_c2y_model.named_parameters():
                norm_factor += np.prod(params.shape)
            for _, params in self.dynamic_c2y_model.named_parameters():
                loss += self.dynamic_weights_reg * torch.norm(
                    params,
                    p=self.dynamic_weights_reg_norm
                ) / norm_factor

        if self.dynamic_activations_reg and (self._coeffs is not None):
            loss += self.dynamic_activations_reg * torch.norm(
                self._coeffs,
                p=self.dynamic_activations_reg_norm,
            ) / np.prod(self._coeffs.shape)
            self._coeffs = None

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
        bottleneck = self._construct_c2y_input(
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            probs=probs,
            **task_loss_kwargs,
        )
        y_logits = self._predict_labels(
            bottleneck,
            latent_space=task_loss_kwargs['latent_space'],
        )
        return self.task_loss_weight * self.loss_task(
            (
                y_logits
                if y_logits.shape[-1] > 1 else
                y_logits.reshape(-1)
            ),
            y,
        )

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
        # Shape (B, 2 * k, m)
        pred_concepts = self._mix(pos_embeddings, neg_embeddings, torch.concat([probs, probs], axis=1))

        return self._make_bottleneck(
            pred_concepts=pred_concepts,
            latent_space=task_loss_kwargs['latent_space'],
            probs=probs,
            train=True,
        )

    def _make_bottleneck(
        self,
        pred_concepts,
        latent_space,
        probs,
        train=False,
    ):
        mixcem_pred_concepts = pred_concepts[:, :self.n_concepts, :]
        dynamic_pred_concepts = pred_concepts[:, self.n_concepts:, :]
        bottlenecks = []
        for used_pred_concepts in [mixcem_pred_concepts, dynamic_pred_concepts]:
            if self.bottleneck_pooling == 'additive':
                bottleneck = torch.zeros(
                    (used_pred_concepts.shape[0], self.emb_size)
                ).to(used_pred_concepts.device)
                for i in range(self.n_concepts):
                    bottleneck += used_pred_concepts[:, i, :]
            elif self.bottleneck_pooling == 'mean':
                bottleneck = torch.mean(used_pred_concepts, axis=1)
            elif self.bottleneck_pooling == 'weighted_mean':
                bottleneck = torch.zeros(
                    (used_pred_concepts.shape[0], self.emb_size)
                ).to(used_pred_concepts.device)
                norm_factor = torch.sum(self.concept_pool_weights)
                for i in range(self.n_concepts):
                    bottleneck += (
                        self.concept_pool_weights[i] * used_pred_concepts[:, i, :]
                    )
                bottleneck = bottleneck / norm_factor
            elif self.bottleneck_pooling == 'concat':
                bottleneck = used_pred_concepts.view(used_pred_concepts.shape[0], -1)
            elif 'per_class_mixing' in self.bottleneck_pooling:
                bottleneck = used_pred_concepts
            else:
                raise ValueError(
                    f'Unsupported bottleneck pooling "{self.bottleneck_pooling}".'
                )
            bottlenecks.append(bottleneck)
        return torch.concat(
            bottlenecks,
            dim=1,
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
        if latent is None:
            pre_c = self.pre_concept_model(x)
            model_latent_space = pre_c
            c_sem = []
            mixcem_pred_concepts = []
            mixcem_pos_embs = []
            mixcem_neg_embs = []

            dynamic_pred_concepts = []
            dynamic_pos_embs = []
            dynamic_neg_embs = []

            for i, concept_emb_generator in enumerate(
                self.concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                concept_latent_space = concept_emb_generator(pre_c)
                if self.shared_emb_generator:
                    model_latent_space = concept_latent_space

                # First build the mixcem's fixed concept embedding vectors
                # [Shape: (B, emb_size)]
                mixcem_pos_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 0, :],
                    dim=0,
                ).expand(concept_latent_space.shape[0], -1)
                mixcem_neg_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 1, :],
                    dim=0,
                ).expand(concept_latent_space.shape[0], -1)

                # Then build the CEM's dynamic concept embedding vectors
                # [Shape: (B, emb_size)]
                # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # BEFORE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                dynamic_emb_input = (
                    concept_latent_space if self.dyn_from_shared_space
                    else pre_c
                )
                dyn_embs = self.dynamic_emb_generators[i](dynamic_emb_input)
                # AFTER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # dyn_embs = self.dynamic_emb_generators[i](pre_c)
                dyn_embs = dyn_embs.view(dyn_embs.shape[0], 2, self.emb_size)
                # END OF CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                dynamic_pos_emb = dyn_embs[:, 0, :]
                dynamic_neg_emb = dyn_embs[:, 1, :]


                # We can now predict the concept probabilities
                prob = self.predict_probs(
                    concept_latent_space=concept_latent_space,
                    mixcem_pos_emb=mixcem_pos_emb,
                    mixcem_neg_embedding=mixcem_neg_emb,
                    dynamic_pos_embedding=dynamic_pos_emb,
                    dynamic_neg_embedding=dynamic_neg_emb,
                    idx=i,
                )

                # [Shape: (B, 1)]
                if len(prob.shape) != 2:
                    prob = torch.unsqueeze(prob, dim=-1)

                c_sem.append(prob)

                # Do the mixing for the global embeddings
                mixcem_mixed_embs = self._mix(mixcem_pos_emb, mixcem_neg_emb, prob)
                mixcem_pos_embs.append(mixcem_pos_emb.unsqueeze(1))
                mixcem_neg_embs.append(mixcem_neg_emb.unsqueeze(1))
                mixcem_pred_concepts.append(
                    torch.unsqueeze(mixcem_mixed_embs, dim=1)
                )

                # Do the mixing for the dynamic embeddings
                dynamic_mixed_embs = self._mix(dynamic_pos_emb, dynamic_neg_emb, prob)
                dynamic_pos_embs.append(dynamic_pos_emb.unsqueeze(1))
                dynamic_neg_embs.append(dynamic_neg_emb.unsqueeze(1))
                dynamic_pred_concepts.append(
                    torch.unsqueeze(dynamic_mixed_embs, dim=1)
                )

            c_sem = torch.cat(c_sem, dim=-1)
            mixcem_pos_embs = torch.cat(mixcem_pos_embs, dim=1)
            mixcem_neg_embs = torch.cat(mixcem_neg_embs, dim=1)
            mixcem_pred_concepts = torch.cat(mixcem_pred_concepts, dim=1)

            dynamic_pos_embs = torch.cat(dynamic_pos_embs, dim=1)
            dynamic_neg_embs = torch.cat(dynamic_neg_embs, dim=1)
            dynamic_pred_concepts = torch.cat(dynamic_pred_concepts, dim=1)


            # Now put everything together
            pred_concepts = torch.concat(
                [mixcem_pred_concepts, dynamic_pred_concepts],
                dim=1,
            )
            pos_embs = torch.concat(
                [mixcem_pos_embs, dynamic_pos_embs],
                dim=1,
            )
            neg_embs = torch.concat(
                [mixcem_neg_embs, dynamic_neg_embs],
                dim=1,
            )
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

    def _predict_labels(self, bottleneck, latent_space, **task_loss_kwargs):
        if len(bottleneck.shape) == 2:
            bottleneck = bottleneck.view(bottleneck.shape[0], 2*self.n_concepts, self.emb_size)

        mixcem_bottleneck = bottleneck[:, :self.n_concepts, :]
        dynamic_bottleneck = bottleneck[:, self.n_concepts:, :]
        if 'per_class_mixing' in self.bottleneck_pooling:
            # Shape (B, k, m)
            mixcem_concept_vectors = torch.nn.functional.leaky_relu(
                mixcem_bottleneck
            )
            dynamic_concept_vectors = torch.nn.functional.leaky_relu(
                dynamic_bottleneck
            )

            # self.downstream_label_weights has shape (L, k)
            # We want shape (B, L, m)
            mixcem_mixed_emb = torch.einsum(
                'bkm,lk->blm',
                mixcem_concept_vectors,
                self.mixcem_downstream_label_weights
            )
            dynamic_mixed_emb = torch.einsum(
                'bkm,lk->blm',
                dynamic_concept_vectors,
                self.dynamic_downstream_label_weights
            )

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
                y_pred = self.c2y_model(
                    mixcem_bottleneck=mixcem_mixed_emb,
                    dynamic_bottleneck=dynamic_mixed_emb,
                    latent_space=latent_space,
                ).squeeze(-1)
        else:
            y_pred = self.c2y_model(
                mixcem_bottleneck=mixcem_bottleneck.view(mixcem_bottleneck.shape[0], -1),
                dynamic_bottleneck=dynamic_bottleneck.view(dynamic_bottleneck.shape[0], -1),
                latent_space=latent_space,
            )
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

        y_pred = self._predict_labels(bottleneck, latent_space=latent_space)

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


    def freeze_pred_mixture_model(self, freeze_emb_generators=False):
        for param in self.pred_mixture_model.parameters():
            param.requires_grad = False

    def unfreeze_pred_mixture_model(self):
        for param in self.pred_mixture_model.parameters():
            param.requires_grad = False




    def freeze_concept_embeddings(self):
        self.concept_embeddings.requires_grad = False
        self.contrastive_scale.requires_grad = False

    def unfreeze_concept_embeddings(self):
        self.concept_embeddings.requires_grad = True
        self.contrastive_scale.requires_grad = True

    def freeze_mixcem_model(self, freeze_concept_emb_generators=False):
        self.freeze_concept_embeddings()
        self.freeze_mixcem_label_predictor()
        if freeze_concept_emb_generators:
            for emb_generator in self.concept_emb_generators:
                for param in emb_generator.parameters():
                    param.requires_grad = False

    def unfreeze_mixcem_model(self):
        self.unfreeze_concept_embeddings()
        self.unfreeze_mixcem_label_predictor()
        for emb_generator in self.concept_emb_generators:
            for param in emb_generator.parameters():
                param.requires_grad = True

    def freeze_mixcem_label_predictor(self):
        if 'per_class_mixing' in self.bottleneck_pooling:
            for submodel in self.mixcem_c2y_model:
                for param in submodel.parameters():
                    param.requires_grad = False
            self.mixcem_downstream_label_weights.requires_grad = False
            if self.bottleneck_pooling == 'per_class_mixing_shared':
                self.class_embeddings.requires_grad = False
        else:
            for param in self.mixcem_c2y_model.parameters():
                param.requires_grad = False

    def unfreeze_mixcem_label_predictor(self):
        if 'per_class_mixing' in self.bottleneck_pooling:
            for submodel in self.mixcem_c2y_model:
                for param in submodel.parameters():
                    param.requires_grad = True
            self.mixcem_downstream_label_weights.requires_grad = True
            if self.bottleneck_pooling == 'per_class_mixing_shared':
                self.mixcem_class_embeddings.requires_grad = True
        else:
            for param in self.mixcem_c2y_model.parameters():
                param.requires_grad = True




    def freeze_dynamic_model(self):
        self.freeze_dynamic_label_predictor()
        self.freeze_dynamic_prob_generators()

    def unfreeze_dynamic_model(self):
        self.unfreeze_dynamic_label_predictor()
        self.unfreeze_dynamic_prob_generators()

    def freeze_dynamic_prob_generators(self):
        for submodel in self.dynamic_emb_generators:
            for param in submodel.parameters():
                param.requires_grad = False
        for submodel in self._dynamic_prob_score_models:
            for param in submodel.parameters():
                param.requires_grad = False

    def unfreeze_dynamic_prob_generators(self):
        for submodel in self.dynamic_emb_generators:
            for param in submodel.parameters():
                param.requires_grad = True
        for submodel in self._dynamic_prob_score_models:
            for param in submodel.parameters():
                param.requires_grad = True

    def freeze_dynamic_label_predictor(self):
        if 'per_class_mixing' in self.bottleneck_pooling:
            for param in self.dynamic_c2y_model.parameters():
                param.requires_grad = False
            self.dynamic_downstream_label_weights.requires_grad = False
            if self.bottleneck_pooling == 'per_class_mixing_shared':
                self.dynamic_class_embeddings.requires_grad = False
        else:
            for param in self.dynamic_c2y_model.parameters():
                param.requires_grad = False

    def unfreeze_dynamic_label_predictor(self):
        if 'per_class_mixing' in self.bottleneck_pooling:
            for param in self.dynamic_c2y_model.parameters():
                param.requires_grad = True
            self.dynamic_downstream_label_weights.requires_grad = True
            if self.bottleneck_pooling == 'per_class_mixing_shared':
                self.dynamic_class_embeddings.requires_grad = True
        else:
            for param in self.dynamic_c2y_model.parameters():
                param.requires_grad = True