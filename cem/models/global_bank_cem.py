import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch

from cem.models.cbm import compute_accuracy
from cem.models.intcbm import IntAwareConceptEmbeddingModel
from torchvision.models import resnet50
import cem.train.utils as utils


class GlobalBankConceptEmbeddingModel(IntAwareConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        n_concept_variants=5,
        tempterature=10,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation=None,
        concept_loss_weight=1,
        task_loss_weight=1,
        intervention_task_discount=1,

        # Mixing
        bottleneck_pooling='concat',
        selection_mode='average_dist',
        distance_selection='hard',
        soft_select=True,
        learnable_prob=False,
        remap_context=False,

        # dynamic embedding
        add_dynamic_embedding=False,
        dynamic_emb_concept_loss_weight=0,

        fixed_embeddings=False,
        initial_concept_embeddings=None,
        fixed_scale=None,

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

        shared_emb_generator=True,

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

        **kwargs,
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        if bottleneck_pooling == 'pcm':
            bottleneck_pooling = 'per_class_mixing'
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
        if c_extractor_arch == "identity":
            self.pre_concept_model = lambda x: x
        else:
            self.pre_concept_model = c_extractor_arch(output_dim=None)
        if self.pre_concept_model == "identity":
            c_extractor_arch = "identity"
            self.pre_concept_model = lambda x: x


        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent

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
        self.n_concept_variants = n_concept_variants
        self.distance_selection = distance_selection
        if selection_mode == 'hierarchical':
            self.hierarchical_concepts = torch.nn.Parameter(
                torch.rand(
                    self.n_concepts,
                    n_concept_variants,
                    emb_size,
                ),
                requires_grad=(not fixed_embeddings),
            )
            n_in_feats = list(
                self.pre_concept_model.modules()
            )[-1].out_features
            self.hierarchical_selector = torch.nn.ModuleList([
                torch.nn.Sequential(*[
                    torch.nn.BatchNorm1d(num_features=(emb_size + n_in_feats)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(
                        (emb_size + n_in_feats),
                        emb_size,
                    ),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(
                        emb_size,
                        self.n_concept_variants,
                    ),
                ])
                for _ in range(n_concepts)
            ])
            n_concept_variants = 1
        if (initial_concept_embeddings is False) or (
            initial_concept_embeddings is None
        ):
            initial_concept_embeddings = torch.rand(
                self.n_concepts,
                n_concept_variants,
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
        if selection_mode == 'average_dist':
            if fixed_scale is not None:
                self.contrastive_scale = torch.nn.Parameter(
                    fixed_scale * torch.ones((self.n_concepts, n_concept_variants + int(add_dynamic_embedding))),
                    requires_grad=False,
                )
            else:
                self.contrastive_scale = torch.nn.Parameter(
                    torch.rand((self.n_concepts, n_concept_variants + int(add_dynamic_embedding))),
                    requires_grad=True,
                )
        else:
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
                            torch.nn.BatchNorm1d(num_features=n_in_feats),
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
                        torch.nn.BatchNorm1d(num_features=n_in_feats),
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

        if c2y_model is None:
            # Else we construct it here directly
            if 'per_class_mixing' in self.bottleneck_pooling:
                self.downstream_label_weights = torch.nn.Parameter(
                    torch.rand((n_tasks, self.n_concepts)),
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
            torch.nn.CrossEntropyLoss(weight=task_class_weights, reduction='mean')
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights,
                reduction='mean',
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
        self.tempterature = tempterature

        self._cos_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

        self.selection_mode = selection_mode
        self.soft_select = soft_select
        self.learnable_prob = learnable_prob
        self.remap_context = remap_context
        self.add_dynamic_embedding = add_dynamic_embedding
        self.dynamic_emb_concept_loss_weight = dynamic_emb_concept_loss_weight
        self.dynamic_embedding_models = None
        self._dyn_embs = None
        if add_dynamic_embedding:
            assert selection_mode != 'hierarchical', 'Unsupported as of now'
            self.dynamic_embedding_models = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(
                        self.emb_size,
                        self.emb_size,
                    ),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(
                        self.emb_size,
                        2*self.emb_size,
                    ),
                )
                for _ in range(self.n_concepts)
            ])
            self.n_concept_variants += 1
            if self.selection_mode != 'average_dist' and (
                dynamic_emb_concept_loss_weight
            ):
                self._dyn_embs = {}
                self._dyn_prob_model = torch.nn.Linear(
                    2 * emb_size,
                    1,
                )
        if selection_mode in ['max_distance', 'min_distance']:
            if self.learnable_prob:
                raise ValueError(
                    f'We do not support learnable_prob yet when dealing with max_distance mode!'
                )
        elif selection_mode == 'hierarchical':
            pass
        elif selection_mode == 'attention':
            # self.selection_models = torch.nn.ModuleList([
            #     torch.nn.Linear(
            #         emb_size,
            #         self.n_concept_variants,
            #     )
            #     for i in range(self.n_concepts)
            # ])
            self.selection_models = torch.nn.ModuleList([
               torch.nn.MultiheadAttention(
                    self.emb_size,
                    1,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(self.n_concepts)
            ])
        elif selection_mode == 'learnt':
            assert learnable_prob, 'learnt selection mode implies that the probability module is learnt!'
            self.concept_context_generators = torch.nn.ModuleList()
            for i in range(n_concepts):
                if embedding_activation is None:
                    act_to_use = []
                elif embedding_activation == "sigmoid":
                    act_to_use = [torch.nn.Sigmoid()]
                elif embedding_activation == "leakyrelu":
                    act_to_use = [torch.nn.LeakyReLU()]
                elif embedding_activation == "relu":
                    act_to_use = [torch.nn.ReLU()]
                else:
                    raise ValueError(
                        f'Unsupported embedding activation "{embedding_activation}"'
                    )
                self.concept_context_generators.append(
                    torch.nn.Sequential(*([
                        torch.nn.Linear(
                            emb_size,
                            2*emb_size,
                        ),
                    ] + act_to_use))
                )
        elif selection_mode == 'average_dist':
            if add_dynamic_embedding:
                self.dynamic_prob_model = torch.nn.Linear(
                    2 * emb_size,
                    1,
                )

        else:
            raise ValueError(
                f'Unsupported selection mode "{selection_mode}"'
            )

        if self.learnable_prob:
            self.prob_model = torch.nn.Linear(
                2 * emb_size,
                1,
            )
            if self.remap_context:
                self.remap_models = torch.nn.ModuleList([
                    torch.nn.Linear(
                        2*self.emb_size,
                        2*self.emb_size,
                    )
                    for _ in range(self.n_concepts)
                ])
        else:
            self.prob_model = None

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
        if self.add_dynamic_embedding and (self._dyn_embs is not None) and (
            self.dynamic_emb_concept_loss_weight
        ):
            c_preds = torch.concat(
                [
                    self.sig(self._dyn_prob_model(self._dyn_embs[i]))
                    for i in range(self.n_concepts)
                ],
                dim=1,
            )
            loss += self._compute_concept_loss(
                c=c,
                c_pred=c_preds,
            )
            # And reset them
            self._dyn_embs = {}
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
        y_logits = self._predict_labels(bottleneck)
        return self.task_loss_weight * self.loss_task(
            (
                y_logits
                if y_logits.shape[-1] > 1 else
                y_logits.reshape(-1)
            ),
            y,
        )

    def _mix(self, pos_embs, neg_embs, probs, pre_concepts_latent_space):
        if len(probs.shape) != len(pos_embs.shape):
            probs = probs.unsqueeze(-1)

        mixed = probs * pos_embs + (1 - probs) * neg_embs
        if self.selection_mode == 'hierarchical':
            new_mixed = []
            for concept_idx in range(self.n_concepts):
                # Shape (B, n_concept_variants)
                hierarchical_selection = self.hierarchical_selector[concept_idx](
                    torch.concat([mixed[:, concept_idx, :], pre_concepts_latent_space], dim=-1)
                )
                # Shape (B, n_concept_variants, 1)
                hierarchical_selection = torch.softmax(
                    self.tempterature * hierarchical_selection,
                    dim=-1,
                ).unsqueeze(-1)
                #  self.hierarchical_concepts has shape (n_concepts, n_concept_variants, emb_size)
                #  Each new element in this list needs to have shape (B, 1, emb_size)
                new_mixed.append((
                    hierarchical_selection * self.hierarchical_concepts[concept_idx:concept_idx+1, :, :]
                ).sum(1).unsqueeze(1))
            mixed = torch.concat(new_mixed, dim=1)
            # We get size (B, n_concepts, emb_size)
        return mixed


    def _construct_c2y_input(
        self,
        pos_embeddings,
        neg_embeddings,
        probs,
        **task_loss_kwargs,
    ):
        pred_concepts = self._mix(
            pos_embeddings,
            neg_embeddings,
            probs,
            pre_concepts_latent_space=task_loss_kwargs['pre_concepts_latent_space'],
        )
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
        elif 'per_class_mixing' in self.bottleneck_pooling:
            bottleneck = pred_concepts
        else:
            raise ValueError(
                f'Unsupported bottleneck pooling "{self.bottleneck_pooling}".'
            )
        return bottleneck


    def _contrastive_loss(self, neg_distance, pos_distance):
        return neg_distance - pos_distance

    def _maximum(self, values):
        distr = torch.softmax(
            self.tempterature*values,
            dim=-1,
        )
        return (values*distr).sum(-1), torch.argmax(distr, dim=-1)

    def _get_concept_embedding(
        self,
        concept_latent_space,
        concept_idx,
        variant_idx,
    ):
        if variant_idx < self.concept_embeddings.shape[1]:
            pos_symbolic_vector = torch.unsqueeze(
                self.concept_embeddings[concept_idx, variant_idx, 0, :],
                dim=0,
            ).expand(concept_latent_space.shape[0], -1)
            neg_symbolic_vector = torch.unsqueeze(
                self.concept_embeddings[concept_idx, variant_idx, 1, :],
                dim=0,
            ).expand(concept_latent_space.shape[0], -1)
        else:
            assert self.dynamic_embedding_models is not None
            assert variant_idx == self.concept_embeddings.shape[1]
            context = self.dynamic_embedding_models[concept_idx](concept_latent_space)
            pos_symbolic_vector = context[:, :self.emb_size]
            neg_symbolic_vector = context[:, self.emb_size:]
            if self._dyn_embs is not None:
                self._dyn_embs[concept_idx] = torch.concat(
                    [pos_symbolic_vector, neg_symbolic_vector],
                    dim=-1,
                )
        return pos_symbolic_vector, neg_symbolic_vector

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
            concept_latent_space = None
            # First predict all the concept probabilities
            for concept_idx, concept_emb_generator in enumerate(
                self.concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                if (not self.shared_emb_generator) or (
                    concept_latent_space is None
                ):
                    concept_latent_space = concept_emb_generator(pre_c)
                if self.shared_emb_generator:
                    model_latent_space = concept_latent_space

                # Now let's compute the distances between all positive
                # concept anchors and all negative concept anchors
                # [Shape: (B, emb_size)]
                if self.selection_mode in ['max_distance', 'min_distance']:
                    pos_anchor_distances = []
                    neg_anchor_distances = []
                    for variant_idx in range(self.n_concept_variants):
                        pos_symbolic_vector, neg_symbolic_vector = self._get_concept_embedding(
                            concept_latent_space=concept_latent_space,
                            concept_idx=concept_idx,
                            variant_idx=variant_idx,
                        )
                        pos_anchor_distances.append(
                            (pos_symbolic_vector - concept_latent_space).pow(2).sum(-1).sqrt().unsqueeze(-1)
                        )
                        neg_anchor_distances.append(
                            (neg_symbolic_vector - concept_latent_space).pow(2).sum(-1).sqrt().unsqueeze(-1)
                        )
                    # Shape (B, n_concept_variants)
                    pos_anchor_distances = torch.concat(pos_anchor_distances, dim=-1)
                    neg_anchor_distances = torch.concat(neg_anchor_distances, dim=-1)
                    if self.selection_mode == 'min_distance':
                        pos_anchor_distances = -pos_anchor_distances
                        neg_anchor_distances = -neg_anchor_distances

                    pos_distance, max_pos_idx = self._maximum(pos_anchor_distances)
                    neg_distance, max_neg_idx = self._maximum(neg_anchor_distances)

                    if self.selection_mode == 'min_distance':
                        pos_distance = -pos_distance
                        neg_distance = -neg_distance

                    # Now determine which concept embedding was the closest so that
                    # we know which positive and negative embedding to use!
                    prob = self.sig(
                        self.contrastive_scale[concept_idx] * self._contrastive_loss(
                            neg_distance=neg_distance,
                            pos_distance=pos_distance,
                        )
                    )
                    if self.distance_selection == 'hard':
                        # selected_idxs = torch.where(prob >= 0.5, max_pos_idx, max_neg_idx)
                        # We can now get the actual embeddings
                        selected_pos_embs = self.concept_embeddings[concept_idx, max_pos_idx, 0, :]
                        selected_neg_embs = self.concept_embeddings[concept_idx, max_neg_idx, 1, :]
                    elif self.distance_selection == 'soft':
                        pos_distr = torch.softmax(
                            self.tempterature * pos_anchor_distances,
                            dim=-1,
                        ).unsqueeze(-1)
                        selected_pos_embs = (
                            self.concept_embeddings[concept_idx:concept_idx+1, :, 0, :] *
                            pos_distr
                        ).sum(1)

                        neg_distr = torch.softmax(
                            self.tempterature * neg_anchor_distances,
                            dim=-1,
                        ).unsqueeze(-1)
                        selected_neg_embs = (
                            self.concept_embeddings[concept_idx:concept_idx+1, :, 1, :] *
                            neg_distr
                        ).sum(1)
                    elif self.distance_selection == 'stochastic':
                        if not training:
                            pos_distr = torch.nn.functional.gumbel_softmax(
                                pos_anchor_distances,
                                tau=100,
                                hard=True,
                                eps=1e-10,
                                dim=-1,
                            ).unsqueeze(-1)
                        else:
                            pos_distr = torch.nn.functional.gumbel_softmax(
                                pos_anchor_distances,
                                tau=self.tempterature,
                                hard=(not self.soft_select),
                                eps=1e-10,
                                dim=-1,
                            ).unsqueeze(-1)
                        selected_pos_embs = (
                            self.concept_embeddings[concept_idx:concept_idx+1, :, 0, :] *
                            pos_distr
                        ).sum(1)

                        if not training:
                            neg_distr = torch.nn.functional.softmax(
                                self.tempterature*neg_anchor_distances,
                                dim=-1,
                            ).unsqueeze(-1)
                        else:
                            neg_distr = torch.nn.functional.gumbel_softmax(
                                neg_anchor_distances,
                                tau=self.tempterature,
                                hard=(not self.soft_select),
                                eps=1e-10,
                                dim=-1,
                            ).unsqueeze(-1)
                        selected_neg_embs = (
                            self.concept_embeddings[concept_idx:concept_idx+1, :, 1, :] *
                            neg_distr
                        ).sum(1)
                    else:
                        raise ValueError(
                            f'Unsupported distance_selection '
                            f'"{self.distance_selection}"'
                        )


                elif self.selection_mode == 'attention':
                    selection_model = self.selection_models[concept_idx]
                    # query = selection_model(concept_latent_space)
                    # # attention = torch.nn.functional.softmax(
                    # #     self.tempterature*query,
                    # #     dim=-1,
                    # # ).unsqueeze(-1)
                    # if not training:
                    #     attention = torch.nn.functional.softmax(
                    #         self.tempterature*query,
                    #         dim=-1,
                    #     ).unsqueeze(-1)
                    # else:
                    #     attention = torch.nn.functional.gumbel_softmax(
                    #         query,
                    #         tau=self.tempterature,
                    #         hard=(not self.soft_select),
                    #         eps=1e-10,
                    #         dim=-1,
                    #     ).unsqueeze(-1)

                    # if not self.soft_select:
                    #     raise NotImplementedError('hard select for attention')
                    #     # attention = torch.max(attention, dim=-1, keepdim=True)
                    # selected_pos_embs = (self.concept_embeddings[concept_idx, :, 0, :].unsqueeze(0) * attention).sum(1)
                    # selected_neg_embs = (self.concept_embeddings[concept_idx, :, 1, :].unsqueeze(0) * attention).sum(1)

                    pos_keys = []
                    neg_keys = []
                    for variant_idx in range(self.n_concept_variants):
                        pos_symbolic_vector, neg_symbolic_vector = self._get_concept_embedding(
                            concept_latent_space=concept_latent_space,
                            concept_idx=concept_idx,
                            variant_idx=variant_idx,
                        )
                        pos_keys.append(pos_symbolic_vector.unsqueeze(1))
                        neg_keys.append(neg_symbolic_vector.unsqueeze(1))
                    pos_keys = torch.concat(pos_keys, axis=1)
                    neg_keys = torch.concat(neg_keys, axis=1)
                    selected_pos_embs, attn_output_weights = selection_model(
                        query=concept_latent_space.unsqueeze(1),
                        key=pos_keys,
                        value=pos_keys,
                    )
                    selected_pos_embs = selected_pos_embs.squeeze(1)

                    selected_neg_embs, attn_output_weights = selection_model(
                        query=concept_latent_space.unsqueeze(1),
                        key=neg_keys,
                        value=neg_keys,
                    )
                    selected_neg_embs = selected_neg_embs.squeeze(1)

                    if self.remap_context:
                        new_context = self.remap_models[concept_idx](
                            torch.concat(
                                [selected_pos_embs, selected_neg_embs],
                                dim=-1,
                            )
                        )
                        selected_pos_embs = new_context[:, :self.emb_size]
                        selected_neg_embs = new_context[:, self.emb_size:]

                    if self.prob_model is None:
                        pos_distance = (
                            selected_pos_embs - concept_latent_space
                        ).pow(2).sum(-1).sqrt().unsqueeze(-1)
                        neg_distance = (
                            selected_neg_embs - concept_latent_space
                        ).pow(2).sum(-1).sqrt().unsqueeze(-1)

                        prob = self.sig(
                            self.contrastive_scale[concept_idx] * self._contrastive_loss(
                                neg_distance=neg_distance,
                                pos_distance=pos_distance,
                            )
                        )
                    else:
                        prob_logits = self.prob_model(torch.concat(
                            [selected_pos_embs, selected_neg_embs],
                            dim=-1,
                        ))
                        prob = self.sig(
                            prob_logits
                        )

                elif self.selection_mode == 'learnt':
                    context = self.concept_context_generators[concept_idx](
                        concept_latent_space
                    )
                    prob = self.sig(self.prob_model(context))
                    selected_pos_embs = context[:, :self.emb_size]
                    selected_neg_embs = context[:, self.emb_size:]


                elif self.selection_mode == 'average_dist':
                    pos_variant_embs = []
                    neg_variant_embs = []
                    variant_probs = []
                    for variant_idx in range(self.n_concept_variants):
                        pos_symbolic_vector, neg_symbolic_vector = self._get_concept_embedding(
                            concept_latent_space=concept_latent_space,
                            concept_idx=concept_idx,
                            variant_idx=variant_idx,
                        )
                        if variant_idx < self.concept_embeddings.shape[1]:
                            pos_dist = (
                                pos_symbolic_vector - concept_latent_space
                            ).pow(2).sum(-1).sqrt().unsqueeze(-1)
                            neg_dist = (
                                neg_symbolic_vector - concept_latent_space
                            ).pow(2).sum(-1).sqrt().unsqueeze(-1)

                            variant_probs.append(self.sig(
                                self.contrastive_scale[concept_idx, variant_idx] * self._contrastive_loss(
                                    neg_distance=neg_dist,
                                    pos_distance=pos_dist,
                                )
                            ))
                        else:
                            variant_probs.append(self.sig(
                                self.dynamic_prob_model(torch.concat(
                                    [pos_symbolic_vector, neg_symbolic_vector],
                                    dim=-1,
                                ))
                            ))
                        pos_variant_embs.append(pos_symbolic_vector.unsqueeze(1))
                        neg_variant_embs.append(neg_symbolic_vector.unsqueeze(1))
                    # Shape (B, n_variants, emb_size)
                    pos_variant_embs = torch.concat(pos_variant_embs, axis=1)
                    neg_variant_embs = torch.concat(neg_variant_embs, axis=1)
                    # Shape (B, n_variants, 1)
                    variant_probs = torch.concat(variant_probs, axis=1).unsqueeze(-1)

                    # Shape (B, emb_size)
                    selected_pos_embs = pos_variant_embs.mean(1) #(pos_variant_embs * variant_probs).sum(1)
                    selected_neg_embs = neg_variant_embs.mean(1) #(neg_variant_embs * variant_probs).sum(1)

                    if self.prob_model is None:
                        prob = torch.mean(variant_probs.squeeze(-1), dim=-1)
                    else:
                        prob_logits = self.prob_model(torch.concat(
                            [selected_pos_embs, selected_neg_embs],
                            dim=-1,
                        ))
                        prob = self.sig(
                            prob_logits
                        )
                elif self.selection_mode == 'hierarchical':
                    selected_pos_embs, selected_neg_embs = self._get_concept_embedding(
                        concept_latent_space=concept_latent_space,
                        concept_idx=concept_idx,
                        variant_idx=0,
                    )
                    pos_dist = (
                        selected_pos_embs - concept_latent_space
                    ).pow(2).sum(-1).sqrt().unsqueeze(-1)
                    neg_dist = (
                        selected_neg_embs - concept_latent_space
                    ).pow(2).sum(-1).sqrt().unsqueeze(-1)

                    prob = self.sig(
                        self.contrastive_scale[concept_idx] * self._contrastive_loss(
                            neg_distance=neg_dist,
                            pos_distance=pos_dist,
                        )
                    )

                # And we can mix them
                # [Shape: (B, 1)]
                if len(prob.shape) != 2:
                    prob = torch.unsqueeze(prob, dim=-1)

                pos_embs.append(selected_pos_embs.unsqueeze(1))
                neg_embs.append(selected_neg_embs.unsqueeze(1))
                c_sem.append(prob)


            c_sem = torch.cat(c_sem, dim=-1)
            pos_embs = torch.cat(pos_embs, dim=1)
            neg_embs = torch.cat(neg_embs, dim=1)
            pred_concepts = self._mix(
                pos_embs,
                neg_embs,
                c_sem,
                pre_concepts_latent_space=pre_c,
            )
            latent = pre_c,  c_sem, pred_concepts, model_latent_space, pos_embs, neg_embs
        else:
            pre_c, c_sem, pred_concepts, model_latent_space, pos_embs, neg_embs = latent

        self._task_loss_kwargs.update({
            'latent_space': model_latent_space,
            'pre_concepts_latent_space': pre_c,
        })
        return c_sem, pos_embs, neg_embs, {
            "pred_concepts": pred_concepts,
            "pre_c": pre_c,
            "latent_space": model_latent_space,
            "latent": latent,
            'pre_concepts_latent_space': pre_c,
        }

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
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

    def _predict_labels(self, bottleneck, **task_loss_kwargs):
        if 'per_class_mixing' in self.bottleneck_pooling:
            # Shape (B, k, m)
            concept_vectors = torch.nn.functional.leaky_relu(bottleneck[:, :self.n_concepts, :])

            # self.downstream_label_weights has shape (L, k)
            # We want shape (B, L, m)
            mixed_emb = torch.einsum('bkm,lk->blm', concept_vectors, self.downstream_label_weights[:, :self.n_concepts])

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