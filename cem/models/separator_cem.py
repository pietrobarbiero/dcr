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

def logit(x):
    return torch.log(x / (1 - x + 1e-8))

class SeparatorConceptEmbeddingModel(IntAwareConceptEmbeddingModel):
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
        attention_fn='softmax',
        ood_dropout_prob=0,
        margin_loss_weight=0,
        separator_warmup_steps=0,
        box_temperature=10,
        bounds_loss_weight=0,
        pooling_mode='concat',
        init_bound_val=5,
        sep_loss_weight=0,
        selection_mode='prob_box',
        projection_dim=None,  # Whether or not we project the embeddings to another dimension before checking for separability
        separator_mode='individual',  # How we generate the global selection mask once we have the scores
    ):
        self._construct_c2y_model = False
        super(SeparatorConceptEmbeddingModel, self).__init__(
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
        self.selection_mode = selection_mode
        self.temperature = temperature
        self.n_concept_variants = n_concept_variants
        self.attention_fn = attention_fn
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
        if self.n_concept_variants > 1:
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
        else:
            self.attn_model = [
                lambda x: torch.ones(x.shape[0], self.n_concept_variants).to(x.device)
                for _ in range(self.n_concepts)
            ]

        self._margins = None
        self._pos_scores = None
        self._neg_scores = None
        self.margin_loss_weight = margin_loss_weight
        # self.separators = torch.nn.ModuleList([
        #     torch.nn.Sequential(
        #         torch.nn.Linear(
        #             emb_size,
        #             1,
        #         ),
        #         torch.nn.Sigmoid(),
        #     )
        #     for _ in range(self.n_concepts)
        # ])
        self.separators = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, self.emb_size),
                torch.ones(n_concepts, self.emb_size),
            ),
            requires_grad=True,
        )
        self.margin_scales = torch.nn.Parameter(
            torch.rand(n_concepts),
            requires_grad=True,
        )
        self.margin_thresholds = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts),
                torch.ones(n_concepts),
            ),
            requires_grad=True,
        )
        self.separator_warmup_steps = separator_warmup_steps

        self.box_temperature = box_temperature
        self.bounds_loss_weight = bounds_loss_weight
        self.left_bounds = torch.nn.Parameter(
            torch.normal(
                -init_bound_val*torch.ones(n_concepts),
                torch.ones(n_concepts),
            ),
            requires_grad=True,
        )
        self.right_bounds = torch.nn.Parameter(
            torch.normal(
                init_bound_val*torch.ones(n_concepts),
                torch.ones(n_concepts),
            ),
            requires_grad=True,
        )

        self.print_counter = 0

        self.pooling_mode = pooling_mode
        if self.pooling_mode == 'concat':
            # Then nothing to do or change here
            if c2y_model is None:
                # Else we construct it here directly
                units = [
                    n_concepts * emb_size
                ] + (c2y_layers or []) + [n_tasks]
                layers = [torch.nn.Flatten(1, -1)]
                for i in range(1, len(units)):
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                    if i != len(units) - 1:
                        layers.append(torch.nn.LeakyReLU())
                self.c2y_model = torch.nn.Sequential(*layers)
            else:
                self.c2y_model = c2y_model
            self.global_c2y_model = self.c2y_model
        elif self.pooling_mode == 'individual_scores':
            self.dynamic_score_models = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(
                        self.emb_size,
                        n_tasks,
                    ),
                )
                for _ in range(self.n_concepts)
            ])
            self.global_score_models = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(
                        self.emb_size,
                        n_tasks,
                    ),
                )
                for _ in range(self.n_concepts)
            ])

            # self.task_mixture_weights = torch.nn.Parameter(
            #     torch.normal(
            #         torch.ones(n_concepts, n_tasks),
            #         torch.ones(n_concepts, n_tasks),
            #     ),
            #     requires_grad=True,
            # )
            self.c2y_model = self._individual_score_mix
            self.global_c2y_model = lambda x: self._individual_score_mix(x, only_global=True)
        else:
            raise ValueError(
                f'Unsupported pooling mode "{self.pooling_mode}"'
            )

        self._bypass_bottleneck = None


        # self.seps = torch.nn.ModuleList([
        #     torch.nn.Sequential(
        #         torch.nn.Linear(
        #             emb_size,
        #             1,
        #         ),
        #         torch.nn.Sigmoid(),
        #     )
        #     for _ in range(self.n_concepts)
        # ])
        self.projection_dim = projection_dim or emb_size
        self.seps = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, self.projection_dim),
                torch.ones(n_concepts, self.projection_dim),
            ),
            requires_grad=True,
        )
        if self.projection_dim:
            self.projection_mat = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts, emb_size, self.projection_dim),
                torch.ones(n_concepts, emb_size, self.projection_dim),
            ),
            requires_grad=True,
        )
        self._sep_scores = None
        self.sep_loss_weight = sep_loss_weight
        self.sep_loss_fn = torch.nn.BCELoss()
        self.sep_score_scales = torch.nn.Parameter(
            torch.normal(
                torch.zeros(n_concepts),
                torch.ones(n_concepts),
            ),
            requires_grad=True,
        )

        self.separator_mode = separator_mode

        if self.n_concept_variants == 1:
            # And overwrite the global embeddings so that they are directly
            # perpendicular to the separating hyperplane
            self.pos_global_emb_scale = torch.nn.Parameter(
                torch.rand(n_concepts) - 0.5,
                requires_grad=True,
            )
            self.neg_global_emb_scale = torch.nn.Parameter(
                torch.rand(n_concepts) - 0.5,
                requires_grad=True,
            )

        self.sep_right_barriers = torch.nn.Parameter(
                init_bound_val * torch.rand(n_concepts),
                requires_grad=True,
            )

    def _project_embs(self, embs, concept_idx):
        if self.projection_dim == self.emb_size:
            return embs
        result = torch.matmul(
            embs.unsqueeze(1),
            self.projection_mat[concept_idx:concept_idx+1, :, :].expand(embs.shape[0], -1, -1)
        ).squeeze(1)
        return result

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
        if self.current_steps.detach().cpu().numpy() <= self.separator_warmup_steps:
            self._bypass_bottleneck = bottleneck

        return bottleneck

    def _individual_score_mix(self, bottleneck, only_global=False):
        # global_selected will have shape (B, n_concepts)
        global_selected = self._combined_global_selected.squeeze(-1)
        if only_global:
            global_selected = torch.ones_like(global_selected)

        output_logits = None
        for concept_idx in range(self.n_concepts):
            global_embs = bottleneck[:, concept_idx, :self.emb_size]
            global_logits = self.global_score_models[concept_idx](global_embs)

            dynamic_embs = bottleneck[:, concept_idx, self.emb_size:]
            dynamic_logits = self.dynamic_score_models[concept_idx](dynamic_embs)

            combined_scores = (
                global_selected[:, concept_idx:concept_idx+1] * global_logits +
                (1 - global_selected[:, concept_idx:concept_idx+1]) * dynamic_logits
            )

            if output_logits is None:
                output_logits = combined_scores
            else:
                output_logits = output_logits + combined_scores

        return output_logits


    def _construct_rank_model_input(self, bottleneck, prev_interventions):
        if self.pooling_mode == 'individual_scores':
            bottleneck = bottleneck[:, :, self.emb_size:]
        cat_inputs = [
            bottleneck.reshape(bottleneck.shape[0], -1),
            prev_interventions,
        ]
        return torch.concat(
            cat_inputs,
            dim=-1,
        )

    def _box_function(self, x, concept_idx=None, left_bound=None, right_bound=None):
        if left_bound is None:
            assert concept_idx is not None
            left_bound = 0.5*torch.sigmoid(self.left_bounds[concept_idx])
        left_barrier = torch.tanh(self.box_temperature * (x - left_bound))
        if right_bound is None:
            assert concept_idx is not None
            right_bound = 0.5 + 0.5*torch.sigmoid(self.right_bounds[concept_idx])
        right_barrier = torch.tanh(self.box_temperature * (x - right_bound))
        return 0.5 * (left_barrier - right_barrier)

    def _left_box_function(self, x, concept_idx, left_bound):
        return torch.sigmoid(self.box_temperature * (x - left_bound))

    def _right_box_function(self, x, concept_idx, right_bound):
        return torch.sigmoid(self.box_temperature * (right_bound - x))

    def _fuzzy_or(self, a, b, sharpness=5):
        # return torch.log(torch.exp(sharpness * a) + torch.exp(sharpness * b))/sharpness
        return a + b - a * b


    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        extra_outputs = {}
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

        prob_contexts = dynamic_contexts
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


        pos_embeddings = []
        neg_embeddings = []
        combined_margins = []
        pos_scores = []
        neg_scores = []
        combined_global_selected = []
        sep_scores = []

        new_global_contexts = []
        for concept_idx in range(self.n_concepts):
            if (
                (training and self.sep_loss_weight) or
                (self.selection_mode == 'distance')
            ):
                sep_plane = torch.nn.functional.normalize(self.seps[concept_idx:concept_idx+1, :].expand(dynamic_contexts.shape[0], -1), dim=-1)
                if self.n_concept_variants == 1:
                    # And overwrite the global embeddings so that they are directly
                    # perpendicular to the separating hyperplane
                    if (concept_idx) == 0 and (not training) and (self.print_counter % 50 == 0): print("torch.exp(self.pos_global_emb_scale[concept_idx]) =", torch.exp(self.pos_global_emb_scale[concept_idx]))
                    if (concept_idx) == 0 and (not training) and (self.print_counter % 50 == 0): print("torch.exp(self.neg_global_emb_scale[concept_idx]) =", torch.exp(self.neg_global_emb_scale[concept_idx]))
                    new_global_context_pos = torch.exp(self.pos_global_emb_scale[concept_idx]) * sep_plane.unsqueeze(1)
                    new_global_context_neg = -torch.exp(self.neg_global_emb_scale[concept_idx]) * sep_plane.unsqueeze(1)
                    new_global_contexts.append(torch.concat(
                        [new_global_context_pos, new_global_context_neg],
                        dim=-1,
                    ))
                    to_use_global_contexts = new_global_contexts[-1].squeeze(1)
                else:
                    to_use_global_contexts = global_contexts[:, concept_idx, :]


                if self.separator_mode in ['softmax', 'gumbel']:
                    pos_sep_samples = torch.concat(
                        [
                            self._project_embs(dynamic_contexts[:, concept_idx, :self.emb_size], concept_idx),
                            self._project_embs(to_use_global_contexts[:, :self.emb_size], concept_idx),
                        ],
                        dim=0,
                    )
                    neg_sep_samples = torch.concat(
                        [
                            self._project_embs(dynamic_contexts[:, concept_idx, self.emb_size:], concept_idx),
                            self._project_embs(to_use_global_contexts[:, self.emb_size:], concept_idx),
                        ],
                        dim=0,
                    )
                else:
                    pos_sep_samples = self._project_embs(dynamic_contexts[:, concept_idx, :self.emb_size], concept_idx)
                    neg_sep_samples = self._project_embs(dynamic_contexts[:, concept_idx, self.emb_size:], concept_idx)
                sep_plane = torch.nn.functional.normalize(self.seps[concept_idx:concept_idx+1, :].expand(pos_sep_samples.shape[0], -1), dim=-1)

                pos_sep_scores = (pos_sep_samples * sep_plane).sum(-1).unsqueeze(-1)
                neg_sep_scores = (neg_sep_samples * sep_plane).sum(-1).unsqueeze(-1)

                sep_scores.append(torch.concat([pos_sep_scores, neg_sep_scores], dim=0))


            if self.selection_mode == 'prob_box':
                if self.current_steps.detach().cpu().numpy() <= self.separator_warmup_steps:
                    if training:
                        self.task_loss_weight = 0.5
                    global_selected = torch.zeros(dynamic_contexts.shape[0], 1, 1).to(dynamic_contexts.device)
                else:
                    if training:
                        self.task_loss_weight = 1
                    global_selected = self._box_function(c_sem[:, concept_idx], concept_idx=concept_idx).unsqueeze(-1).unsqueeze(-1)
                    if concept_idx == 0 and (not training):
                        if self.print_counter % 50 == 0:
                            print(f"\tTotal global embeddings selected for {np.mean((global_selected > 0.5).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(global_selected).detach().cpu().numpy(), "and a min value of", torch.min(global_selected).detach().cpu().numpy(), "when left margin =", 0.5*torch.sigmoid(self.left_bounds[concept_idx]).detach().cpu().numpy(), "and right margin is", 0.5 + 0.5*torch.sigmoid(self.right_bounds[concept_idx]).detach().cpu().numpy())
                            print("\t\tMax prob", torch.max(c_sem[-1]).detach().cpu().numpy(), "and a min prob", torch.min(c_sem[-1]).detach().cpu().numpy())
                        self.print_counter += 1

                    if not training:
                        global_selected = torch.where(
                            global_selected >= 0.5,
                            torch.ones_like(global_selected),
                            torch.zeros_like(global_selected),
                        )
                    else:
                        # Else we may do some forced global concept uses!
                        forced_on = self.ood_dropout(torch.ones_like(global_selected))
                        global_selected = torch.ones_like(global_selected) * forced_on + (1 - forced_on) * global_selected

                combined_global_selected.append(global_selected)
                pos_global_selected = global_selected
                neg_global_selected = global_selected

            elif self.selection_mode.startswith("fixed_"):
                fixed_val = float(self.selection_mode[len("fixed_") + 1:])
                global_selected = fixed_val * torch.ones(dynamic_contexts.shape[0], 1, 1).to(dynamic_contexts.device)
                pos_global_selected = global_selected
                neg_global_selected = global_selected
                combined_global_selected.append(global_selected)

            elif self.selection_mode == 'distance':
                # Then we will use the distance to the separator as a
                # way to determine which pair to select! We will chose the pair
                # with the greatest
                current_sep_scores = sep_scores[-1]
                # First, let's collect the separator scores from all positive,
                # negative, local, and global embeddings
                # These will be arrays with size (B, 1)
                n_samples = dynamic_contexts.shape[0]


                if self.separator_mode in ['softmax', 'gumbel']:
                    pos_dists_dynamic = current_sep_scores[:n_samples, :]
                    pos_dists_global = current_sep_scores[n_samples:2*n_samples, :]
                    neg_dists_dynamic = current_sep_scores[2*n_samples:3*n_samples, :]
                    neg_dists_global = current_sep_scores[3*n_samples:, :]

                    # Compute the total score we want to mazimize for each!
                    # this will be the separation of both embeddings w.r.t. the
                    # linear separator
                    global_dist_scores = pos_dists_global - neg_dists_global

                    # generate a [0, 1] score for each that is proportional to how
                    # distant each pair is and use the score given to the global
                    # pair as the mixing coefficient

                    dynamic_dist_scores = pos_dists_dynamic - neg_dists_dynamic
                    if (concept_idx) == 0 and (self.print_counter % 25 == 0): print("\tmax(global_dist_scores) =", torch.max(global_dist_scores).detach().cpu().numpy())
                    if (concept_idx) == 0 and (self.print_counter % 25 == 0): print("\tmin(global_dist_scores) =", torch.min(global_dist_scores).detach().cpu().numpy())
                    if (concept_idx) == 0 and (self.print_counter % 25 == 0): print("\tmax(dynamic_dist_scores) =", torch.max(dynamic_dist_scores).detach().cpu().numpy())
                    if (concept_idx) == 0 and (self.print_counter % 25 == 0): print("\tmin(dynamic_dist_scores) =", torch.min(dynamic_dist_scores).detach().cpu().numpy())
                    if self.separator_mode == 'softmax':
                        global_selected = torch.softmax(
                            torch.concat([dynamic_dist_scores, global_dist_scores], dim=-1),
                            dim=-1,
                        )[:, 1:].unsqueeze(-1)
                    else:
                        if training:
                            global_selected = torch.nn.functional.gumbel_softmax(
                                torch.concat([dynamic_dist_scores, global_dist_scores], dim=-1),
                                dim=-1,
                                hard=True,
                            )[:, 1:].unsqueeze(-1)
                        else:
                            global_selected = torch.softmax(
                                torch.concat([dynamic_dist_scores, global_dist_scores], dim=-1),
                                dim=-1,
                            )[:, 1:].unsqueeze(-1)
                    pos_global_selected = global_selected
                    neg_global_selected = global_selected

                elif self.separator_mode == 'sigmoid':
                    # Another alternative is just to focus on the dynamic embeddings: if the embeddings are on the wrong side of the separator, then time to use the global ones!
                    pos_dists_dynamic = current_sep_scores[:n_samples, :]
                    neg_dists_dynamic = current_sep_scores[n_samples:, :]
                    global_selected = torch.sigmoid(self.box_temperature*(neg_dists_dynamic - pos_dists_dynamic)).unsqueeze(-1)
                    pos_global_selected = global_selected
                    neg_global_selected = global_selected

                elif self.separator_mode == 'individual':
                    pos_dists_dynamic = current_sep_scores[:n_samples, :]
                    neg_dists_dynamic = current_sep_scores[n_samples:, :]
                    is_neg_dist_in_box = self._left_box_function(neg_dists_dynamic.squeeze(-1), concept_idx=concept_idx, left_bound=0.05).unsqueeze(-1).unsqueeze(-1)
                    is_pos_dist_in_box = self._right_box_function(pos_dists_dynamic.squeeze(-1), concept_idx=concept_idx, right_bound=0.95).unsqueeze(-1).unsqueeze(-1)
                    global_selected = self._fuzzy_or(
                        is_neg_dist_in_box,
                        is_pos_dist_in_box,
                    )
                    pos_global_selected = is_pos_dist_in_box
                    neg_global_selected = is_neg_dist_in_box
                else:
                    raise ValueError(
                        f'Unsupported separator mode "{self.separator_mode}"'
                    )

                if concept_idx == 0:
                    if self.print_counter % 50 == 0:
                        print(f"\tTotal positive global embeddings selected for {np.mean((pos_global_selected > 0.5).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(pos_global_selected).detach().cpu().numpy(), "and a min value of", torch.min(pos_global_selected).detach().cpu().numpy(), "when min pos dists is =", torch.min(pos_dists_dynamic).detach().cpu().numpy(), "and max pos dists is =", torch.max(pos_dists_dynamic).detach().cpu().numpy())
                        print(f"\tTotal negative global embeddings selected for {np.mean((neg_global_selected > 0.5).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(neg_global_selected).detach().cpu().numpy(), "and a min value of", torch.min(neg_global_selected).detach().cpu().numpy(), "when min neg dists is =", torch.min(neg_dists_dynamic).detach().cpu().numpy(), "and max neg dists is =", torch.max(neg_dists_dynamic).detach().cpu().numpy())
                    self.print_counter += 1

                if not training:
                    pos_global_selected = torch.where(
                        pos_global_selected >= 0.5,
                        torch.ones_like(pos_global_selected),
                        torch.zeros_like(pos_global_selected),
                    )
                    neg_global_selected = torch.where(
                        neg_global_selected >= 0.5,
                        torch.ones_like(neg_global_selected),
                        torch.zeros_like(neg_global_selected),
                    )
                else:
                    # Else we may do some forced global concept uses!
                    pos_forced_on = self.ood_dropout(torch.ones_like(pos_global_selected))
                    # mask = torch.bernoulli(
                    #     torch.ones_like(pos_global_selected) * 0.5,
                    # )
                    pos_global_selected = torch.ones_like(pos_global_selected) * pos_forced_on + (1 - pos_forced_on) * pos_global_selected
                    neg_forced_on = self.ood_dropout(torch.ones_like(neg_global_selected))
                    neg_global_selected = torch.ones_like(neg_global_selected) * neg_forced_on + (1 - neg_forced_on) * neg_global_selected

                combined_global_selected.append(global_selected)

            else:
                raise ValueError(
                    f'Unsupported selection mode "{self.selection_mode}"'
                )

            if self.pooling_mode == 'concat':
                if (concept_idx % 3 == 0) and not training: print(f"\t(concept {concept_idx}) Total positive global embeddings selected for {np.mean((pos_global_selected > 0.5).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(pos_global_selected).detach().cpu().numpy(), "and a min value of", torch.min(pos_global_selected).detach().cpu().numpy(), "and mean", torch.mean(pos_global_selected).detach().cpu().numpy())
                if (concept_idx % 3 == 0) and not training: print(f"\t(concept {concept_idx}) Total negative global embeddings selected for {np.mean((neg_global_selected > 0.5).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(neg_global_selected).detach().cpu().numpy(), "and a min value of", torch.min(neg_global_selected).detach().cpu().numpy(), "and mean", torch.mean(neg_global_selected).detach().cpu().numpy())
                pos_embeddings.append(
                    pos_global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
                    (1 - pos_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size]
                )
                neg_embeddings.append(
                    neg_global_selected * global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
                    (1 - neg_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:]
                )
            elif self.pooling_mode == 'individual_scores':
                pos_embeddings.append(
                    torch.concat(
                        [
                            global_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                            dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                        ],
                        dim=-1,
                    )
                )
                neg_embeddings.append(
                    torch.concat(
                        [
                            global_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
                            dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
                        ],
                        dim=-1,
                    )
                )
            else:
                raise ValueError(
                    f'Unsupported pooling mode "{self.pooling_mode}"'
                )

        if new_global_contexts:
            global_contexts = torch.concat(new_global_contexts, dim=1)

        pos_embeddings = torch.concat(pos_embeddings, dim=1)
        neg_embeddings = torch.concat(neg_embeddings, dim=1)

        if combined_global_selected and (
            self.pooling_mode == 'individual_scores'
        ):
            self._combined_global_selected = torch.concat(combined_global_selected, dim=1)
        # c_sem = torch.cat(c_sem, axis=-1)
        if training:
            # if combined_margins:
            #     self._margins = torch.concat(combined_margins, dim=-1)
            if pos_scores:
                self._pos_scores = torch.concat(pos_scores, dim=-1)
            if neg_scores:
                self._neg_scores = torch.concat(neg_scores, dim=-1)

            if sep_scores:
                self._sep_scores = torch.concat(sep_scores, dim=1)


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
        # if self.current_steps.detach().cpu().numpy() > self.separator_warmup_steps:
        #     if (self._margins is not None) and self.margin_loss_weight:
        #         loss += self.margin_loss_weight * torch.mean(
        #             (-self._margins).mean(-1)
        #         )
        #         self._margins = None

        if (self._pos_scores is not None) and self.margin_loss_weight:
            # We want to maximize the positive scores
            loss += self.margin_loss_weight * torch.mean(
                (-self._pos_scores).mean(-1)
            )
            self._pos_scores = None

        if (self._neg_scores is not None) and self.margin_loss_weight:
            # And minimize the positive scores
            loss += self.margin_loss_weight * torch.mean(
                (self._neg_scores).mean(-1)
            )
            self._pos_scores = None

        if self.current_steps.detach().cpu().numpy() > self.separator_warmup_steps and (self.selection_mode == 'prob_box'):
            loss += -self.bounds_loss_weight*(0.5 + 0.5*torch.sigmoid(self.right_bounds) - 0.5*torch.sigmoid(self.left_bounds)).mean(-1).mean()
        elif self._bypass_bottleneck is not None:
            global_y_logits = self.global_c2y_model(self._bypass_bottleneck)
            loss += self.task_loss_weight * self.loss_task(
                global_y_logits if global_y_logits.shape[-1] > 1 else global_y_logits.reshape(-1),
                y,
            )
            self._bypass_bottleneck = None

        if (self._sep_scores is not None) and self.sep_loss_weight:
            # We will assume the first half are all scores corresponding
            # to positive embeddings and the second half are all scores
            # corresponding to negative embeddings
            ground_truth_labels = torch.concat(
                [
                    torch.ones(self._sep_scores.shape[0]//2, self.n_concepts),
                    torch.zeros(self._sep_scores.shape[0]//2, self.n_concepts),
                ],
                dim=0,
            ).to(self._sep_scores.device)
            loss += self.sep_loss_weight * self.sep_loss_fn(
                self._box_function(self._sep_scores, left_bound=0, right_bound=torch.exp(self.sep_right_barriers)),
                ground_truth_labels,
            )
            # print("box_values[:10] =", self._box_function(self._sep_scores, left_bound=0, right_bound=torch.exp(self.sep_right_barriers))[:10].detach().cpu().numpy())
            self._sep_scores = None

        if self.n_concept_variants == 1:
            # print("torch.exp(self.sep_right_barriers) =", torch.exp(self.sep_right_barriers).detach().cpu().numpy())
            loss += -self.bounds_loss_weight * (
                self.sig(self.pos_global_emb_scale).mean() +
                self.sig(self.neg_global_emb_scale).mean()
            )
            loss += self.bounds_loss_weight * self.sig(self.sep_right_barriers).mean()

        return loss