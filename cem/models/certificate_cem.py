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



################################################################################
## Same as a above but the distributions are learnt from the dynammic context
################################################################################

class CertificateConceptEmbeddingModel(IntAwareConceptEmbeddingModel):
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
        initial_concept_embeddings=None,
        fixed_embeddings=False,
        ood_dropout_prob=0,
        pooling_mode='concat',
        certificate_loss_weight=0,
        selection_mode='individual_joint',
        hard_eval_selection=None,
        selection_sample=False,
        eval_majority_vote=False,
        mixed_probs=False,
        contrastive_reg=0,
        global_ood_prob=0,
        class_wise_temperature=True,
        init_dyn_temps=1.5,
        init_global_temps=0.5,
        learnable_temps=False,
        global_temp_reg=0,
        max_temperature=1,  # Temperature used when approximating the maximum operator using a logsoftmax function
        inference_dyn_prob=False,  # Whether or not we use the dynamically predicted probability as the outpt probability during inference
    ):
        self._construct_c2y_model = False
        bottleneck_size = 2 * emb_size * n_concepts if pooling_mode == 'combined' else emb_size * n_concepts
        super(CertificateConceptEmbeddingModel, self).__init__(
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
            bottleneck_size=bottleneck_size,
        )
        self.temperature = temperature
        self.ood_dropout_prob = ood_dropout_prob
        self.ood_dropout = torch.nn.Dropout(
            p=(1 - ood_dropout_prob),  # We do 1 - ood_prob as this will be applied to the selection of the global embedding
        )

        # Let's generate the global embeddings we will use
        if (initial_concept_embeddings is False) or (
            initial_concept_embeddings is None
        ):
            initial_concept_embeddings = torch.normal(
                torch.zeros(self.n_concepts, 2, emb_size),
                torch.ones(self.n_concepts, 2, emb_size),
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

        self.print_counter = 0
        self.eval_majority_vote = eval_majority_vote
        self.pooling_mode = pooling_mode
        if self.pooling_mode in ['concat', 'additive', 'sigmoidal_additive']:
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
        elif self.pooling_mode == 'combined':
            # Then nothing to do or change here
            if c2y_model is None:
                # Else we construct it here directly
                units = [
                    2 * n_concepts * emb_size
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
            self._combined_global_selected = None
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

            self.c2y_model = self._individual_score_mix
            self.global_c2y_model = lambda x: self._individual_score_mix(x, only_global=True)

        elif self.pooling_mode == 'individual_scores_shared':
            self._combined_global_selected = None
            self.dynamic_score_models = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(
                        self.emb_size,
                        n_tasks,
                    ),
                )
                for _ in range(self.n_concepts)
            ])
            self.global_score_models = self.dynamic_score_models

            self.c2y_model = self._individual_score_mix
            self.global_c2y_model = lambda x: self._individual_score_mix(x, only_global=True)

        else:
            raise ValueError(
                f'Unsupported pooling mode "{self.pooling_mode}"'
            )


        self.hard_eval_selection = hard_eval_selection
        self.hard_selection_value = None
        self.selection_mode = selection_mode
        self.certificate_loss_weight = certificate_loss_weight
        self.class_wise_temperature = class_wise_temperature

        if self.selection_mode == 'global':
            self.certificate_model = torch.nn.Linear(
                2*self.emb_size,
                n_concepts,
                bias=False,
            )
            self._dynamic_selections = None
        elif self.selection_mode in ['individual', 'individual_joint']:
            self.certificate_model = torch.nn.Linear(
                self.emb_size,
                2*n_concepts,
                bias=False,
            )
            self._pos_dynamic_selections = None
            self._neg_dynamic_selections = None
        elif self.selection_mode.startswith('fixed_'):
            pass
        elif self.selection_mode == 'max_class_confidence':
            self.concept_scales = torch.nn.Parameter(
                torch.rand(n_concepts),
                requires_grad=True,
            )
            self.learnable_temps = learnable_temps
            if class_wise_temperature:
                self.dynamic_logit_temperatures = torch.nn.Parameter(
                    init_dyn_temps * (torch.zeros(n_concepts, n_tasks) if learnable_temps else torch.ones(n_concepts, n_tasks)),
                    requires_grad=learnable_temps,
                )

                self.global_logit_temperatures = torch.nn.Parameter(
                    init_global_temps * (torch.zeros(n_concepts, n_tasks) if learnable_temps else torch.ones(n_concepts, n_tasks)),
                    requires_grad=learnable_temps,
                )
            else:
                self.dynamic_logit_temperatures = torch.nn.Parameter(
                    init_dyn_temps * (torch.zeros(n_concepts) if learnable_temps else torch.ones(n_concepts)),
                    requires_grad=learnable_temps,
                )

                self.global_logit_temperatures = torch.nn.Parameter(
                    init_global_temps * (torch.zeros(n_concepts) if learnable_temps else torch.ones(n_concepts)),
                    requires_grad=learnable_temps,
                )
            self.global_concept_context_generators = torch.nn.Linear(
                list(
                    self.pre_concept_model.modules()
                )[-1].out_features,
                self.emb_size,
            )


        elif self.selection_mode == 'max_concept_confidence':
            self.global_concept_context_generators = torch.nn.ModuleList()
            self.concept_scales = torch.nn.Parameter(
                torch.rand(n_concepts),
                requires_grad=True,
            )
            self.learnable_temps = learnable_temps
            self.dynamic_logit_temperatures = torch.nn.Parameter(
                init_dyn_temps * (torch.zeros(n_concepts) if learnable_temps else torch.ones(n_concepts)),
                requires_grad=learnable_temps,
            )

            self.global_logit_temperatures = torch.nn.Parameter(
                init_global_temps * (torch.zeros(n_concepts) if learnable_temps else torch.ones(n_concepts)),
                requires_grad=learnable_temps,
            )
            for i in range(n_concepts):
                self.global_concept_context_generators.append(
                    torch.nn.Sequential(*([
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            self.emb_size,
                        ),
                    ]))
                )
        else:
            raise ValueError(
                f'Unsupported selection mode "{self.selection_mode}"'
            )
        self.certificate_loss = torch.nn.CrossEntropyLoss()
        self.selection_sample = selection_sample
        self.mixed_probs = mixed_probs
        self.contrastive_reg = contrastive_reg
        self._c_logits_global = None
        self.global_ood_prob = global_ood_prob
        self.global_temp_reg = global_temp_reg
        self.max_temperature = max_temperature
        self.inference_dyn_prob = inference_dyn_prob

    def _predict_labels(self, bottleneck, **task_loss_kwargs):
        out = self.c2y_model(bottleneck)
        # print("torch.max(out) =", torch.max(out).detach().cpu().numpy())
        # print("torch.min(out) =", torch.min(out).detach().cpu().numpy())
        return out

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
        return bottleneck

    def _get_dynamic_temps(self, concept_idx):
        if self.learnable_temps:
            return 1e-8 + torch.exp(self.dynamic_logit_temperatures[concept_idx])
        return 1e-8 + self.dynamic_logit_temperatures[concept_idx]

    def _get_global_temps(self, concept_idx):
        if self.learnable_temps:
            return 1e-8 + torch.exp(self.global_logit_temperatures[concept_idx])
        return 1e-8 + self.global_logit_temperatures[concept_idx]

    def _individual_score_mix(self, bottleneck, only_global=False):
        # global_selected will have shape (B, n_concepts)
        global_selected = self._combined_global_selected.squeeze(-1)
        if only_global:
            global_selected = torch.ones_like(global_selected)

        output_logits = None
        for concept_idx in range(self.n_concepts):
            global_embs = bottleneck[:, concept_idx, :self.emb_size]
            if self.selection_mode == 'max_class_confidence':
                global_logits = self.global_score_models[concept_idx](global_embs) / self._get_global_temps(concept_idx).unsqueeze(0)
            else:
                global_logits = self.global_score_models[concept_idx](global_embs)

            dynamic_embs = bottleneck[:, concept_idx, self.emb_size:]
            if self.selection_mode == 'max_class_confidence':
                dynamic_logits = self.dynamic_score_models[concept_idx](dynamic_embs) / self._get_dynamic_temps(concept_idx).unsqueeze(0)
            else:
                dynamic_logits = self.dynamic_score_models[concept_idx](dynamic_embs)

            combined_scores = (
                global_selected[:, concept_idx:concept_idx+1] * global_logits +
                (1 - global_selected[:, concept_idx:concept_idx+1]) * dynamic_logits
            )
            # print("torch.max(global_logits) =", torch.max(global_logits).detach().cpu().numpy())
            # print("torch.max(dynamic_logits) =", torch.max(dynamic_logits).detach().cpu().numpy())
            # print("torch.max(combined_scores) =", torch.max(combined_scores).detach().cpu().numpy())
            # print("torch.min(combined_scores) =", torch.min(combined_scores).detach().cpu().numpy())

            if output_logits is None:
                output_logits = combined_scores
            else:
                output_logits = output_logits + combined_scores
            # print("torch.max(output_logits) =", torch.max(output_logits).detach().cpu().numpy())

        return output_logits


    def _construct_rank_model_input(self, bottleneck, prev_interventions):
        if self.pooling_mode in ['individual_scores', 'individual_scores_shared']:
            bottleneck = bottleneck[:, :, self.emb_size:]
        cat_inputs = [
            bottleneck.reshape(bottleneck.shape[0], -1),
            prev_interventions,
        ]
        return torch.concat(
            cat_inputs,
            dim=-1,
        )

    def _generate_dynamic_concept(self, pre_c, concept_idx):
        context = self.concept_context_generators[concept_idx](pre_c)
        return context

    def _max(self, x):
        # return torch.max(x, dim=-1, keepdim=True)[0]
        return torch.log(torch.sum(self.max_temperature * torch.exp(x), dim=-1, keepdim=True))

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
            global_shifts = []
            global_emb_center = None

            # First predict all the concept probabilities
            for concept_idx in range(self.n_concepts):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[concept_idx]
                dynamic_context = self._generate_dynamic_concept(pre_c, concept_idx=concept_idx)
                if self.selection_mode in ['individual', 'individual_joint']:
                    # self.certificate_model.weight has shape (2*n_concepts, emb_size)
                    # linear_separators = self.certificate_model.weight
                    # pos_idx = 2 * concept_idx
                    # global_context_pos = linear_separators[pos_idx:pos_idx+1, :].expand(pre_c.shape[0], -1)
                    # neg_idx = pos_idx + 1
                    # global_context_neg = linear_separators[neg_idx:neg_idx+1, :].expand(pre_c.shape[0], -1)
                    global_context_pos = self.concept_embeddings[concept_idx:concept_idx+1, 0, :].expand(pre_c.shape[0], -1)
                    global_context_neg = self.concept_embeddings[concept_idx:concept_idx+1, 1, :].expand(pre_c.shape[0], -1)
                elif self.selection_mode in ['max_class_confidence', 'max_concept_confidence']:
                    if global_emb_center is None:
                        global_emb_center = self.global_concept_context_generators(pre_c)
                    dynamic_context = dynamic_context + torch.concat(
                        [
                            self.concept_embeddings[concept_idx:concept_idx+1, 0, :].expand(pre_c.shape[0], -1),
                            self.concept_embeddings[concept_idx:concept_idx+1, 1, :].expand(pre_c.shape[0], -1),
                        ],
                        dim=-1
                    )
                    global_context_pos = (
                        self.concept_embeddings[concept_idx:concept_idx+1, 0, :].expand(pre_c.shape[0], -1)
                        # + self.sig(global_shift[:, :self.emb_size])
                    )
                    global_context_neg = (
                        self.concept_embeddings[concept_idx:concept_idx+1, 1, :].expand(pre_c.shape[0], -1)
                        # + self.sig(global_shift[:, self.emb_size:])
                    )
                else:
                    global_context_pos = self.concept_embeddings[concept_idx:concept_idx+1, 0, :].expand(pre_c.shape[0], -1)
                    global_context_neg = self.concept_embeddings[concept_idx:concept_idx+1, 1, :].expand(pre_c.shape[0], -1)
                global_context = torch.concat(
                    [global_context_pos, global_context_neg],
                    dim=-1,
                )
                global_contexts.append(torch.unsqueeze(global_context, dim=1))
                dynamic_contexts.append(torch.unsqueeze(dynamic_context, dim=1))

            dynamic_contexts = torch.cat(dynamic_contexts, axis=1)
            global_contexts = torch.cat(global_contexts, axis=1)
            latent = dynamic_contexts, global_contexts
        else:
            dynamic_contexts, global_contexts = latent


        # Now we can compute all the probabilites!
        c_sem = []
        c_logits_dynamic = []
        c_logits_global = []
        c_logits = []

        for concept_idx in range(self.n_concepts):
            # pos_dist = (
            #     (dynamic_contexts[:, concept_idx, :self.emb_size] - self.concept_embeddings[concept_idx:concept_idx+1, 0, :])**2
            # ).sum(-1)
            # neg_dist = (
            #     (dynamic_contexts[:, concept_idx, self.emb_size:] - self.concept_embeddings[concept_idx:concept_idx+1, 1, :])**2
            # ).sum(-1)
            # prob = torch.softmax(torch.concat([-neg_dist.unsqueeze(-1), -pos_dist.unsqueeze(-1)], dim=-1), dim=-1)[:, 1:]

            if self.shared_prob_gen:
                prob_gen = self.concept_prob_generators[0]
            else:
                prob_gen = self.concept_prob_generators[concept_idx]
            dyn_logits = prob_gen(dynamic_contexts[:, concept_idx, :])
            c_logits.append(dyn_logits)
            if self.mixed_probs or self.contrastive_reg or (self.selection_mode == 'max_concept_confidence'):
                pos_anchor = self.concept_embeddings[concept_idx:concept_idx+1, 0, :].expand(global_emb_center.shape[0], -1)
                neg_anchor = self.concept_embeddings[concept_idx:concept_idx+1, 1, :].expand(global_emb_center.shape[0], -1)
                neg_dist = (neg_anchor - global_emb_center).pow(2).sum(-1).sqrt()
                pos_dist = (pos_anchor - global_emb_center).pow(2).sum(-1).sqrt()
                global_logits = self.concept_scales[concept_idx] * (neg_dist - pos_dist).unsqueeze(-1)
                if self.selection_mode == 'max_concept_confidence':
                    dyn_logits = dyn_logits / self._get_dynamic_temps(concept_idx)
                    global_logits = global_logits / self._get_global_temps(concept_idx)

                c_logits_dynamic.append(dyn_logits)
                c_logits_global.append(global_logits)

                dyn_prob = self.sig(dyn_logits)
                global_prob = self.sig(global_logits)
            if self.mixed_probs:
                if training or (not self.inference_dyn_prob):
                    prob = (
                        0.5 * dyn_prob +
                        0.5 * global_prob
                    )
                else:
                    prob = dyn_prob
            else:
                prob = self.sig(c_logits[-1])
            c_sem.append(prob)
        c_sem = torch.cat(c_sem, axis=-1)
        # print("torch.max(c_sem) =", torch.max(c_sem).detach().cpu().numpy())
        # print("torch.min(c_sem) =", torch.min(c_sem).detach().cpu().numpy())
        if c_logits:
            c_logits = torch.cat(c_logits, axis=-1)

        pos_embeddings = []
        neg_embeddings = []
        dynamic_selections = []
        pos_dynamic_selections = []
        neg_dynamic_selections = []
        combined_global_selected = []
        if self.contrastive_reg:
            self._c_logits_global = c_logits_global
        for concept_idx in range(self.n_concepts):
            pos_idx = concept_idx
            neg_idx = concept_idx

            if self.selection_mode == 'global':
                selection_outputs = self.certificate_model(
                    dynamic_contexts[:, concept_idx, :]
                )
                selection_probs = torch.softmax(
                    self.temperature * selection_outputs,
                    dim=-1,
                )
                global_selected = 1 - selection_probs[:, concept_idx:concept_idx+1].unsqueeze(1)
                if training and self.global_ood_prob:
                    mask = torch.bernoulli(
                        torch.ones(global_selected.shape[0], 1, 1) * self.global_ood_prob,
                    ).to(global_selected.device)
                    global_selected = torch.where(
                        mask == 1,
                        torch.ones_like(global_selected),
                        global_selected,
                    )
                elif training:
                    forced_on = self.ood_dropout(torch.ones_like(global_selected))
                    global_selected = torch.ones_like(global_selected) * forced_on + (1 - forced_on) * global_selected

                pos_selection_probs = selection_probs
                pos_global_selected = global_selected
                neg_selection_probs = selection_probs
                neg_global_selected = global_selected
                pos_idx = neg_idx = concept_idx
                if self.certificate_loss_weight:
                    dynamic_selections.append(
                        selection_outputs.unsqueeze(1)
                    )
                if self.pooling_mode in ['individual_scores', 'individual_scores_shared']:
                    combined_global_selected.append(global_selected)
            elif self.selection_mode == 'individual':
                pos_selection_outputs = self.certificate_model(
                    dynamic_contexts[:, concept_idx, :self.emb_size]
                )
                pos_selection_probs = torch.softmax(
                    self.temperature * pos_selection_outputs,
                    dim=-1,
                )
                pos_idx = 2*concept_idx
                pos_global_selected = 1 - pos_selection_probs[:, pos_idx:pos_idx+1].unsqueeze(1)

                neg_selection_outputs = self.certificate_model(
                    dynamic_contexts[:, concept_idx, self.emb_size:]
                )
                neg_selection_probs = torch.softmax(
                    self.temperature * neg_selection_outputs,
                    dim=-1,
                )
                neg_idx = pos_idx + 1
                neg_global_selected = 1 - neg_selection_probs[:, neg_idx:neg_idx+1].unsqueeze(1)


                if self.certificate_loss_weight:
                    pos_dynamic_selections.append(
                        pos_selection_outputs.unsqueeze(1)
                    )
                    neg_dynamic_selections.append(
                        neg_selection_outputs.unsqueeze(1)
                    )

                if training:
                    pos_forced_on = self.ood_dropout(torch.ones_like(pos_global_selected))
                    pos_global_selected = torch.ones_like(pos_global_selected) * pos_forced_on + (1 - pos_forced_on) * pos_global_selected
                    neg_forced_on = pos_forced_on #self.ood_dropout(torch.ones_like(neg_global_selected))
                    neg_global_selected = torch.ones_like(neg_global_selected) * neg_forced_on + (1 - neg_forced_on) * neg_global_selected

                if self.pooling_mode in ['individual_scores', 'individual_scores_shared']:
                    global_selected = torch.maximum(
                        pos_global_selected,
                        neg_global_selected,
                    )
                    combined_global_selected.append(global_selected)
            elif self.selection_mode == 'individual_joint':
                pos_selection_outputs = self.certificate_model(
                    dynamic_contexts[:, concept_idx, :self.emb_size]
                )
                pos_selection_probs = torch.softmax(
                    self.temperature * pos_selection_outputs,
                    dim=-1,
                )
                pos_idx = 2*concept_idx

                neg_selection_outputs = self.certificate_model(
                    dynamic_contexts[:, concept_idx, self.emb_size:]
                )
                neg_selection_probs = torch.softmax(
                    self.temperature * neg_selection_outputs,
                    dim=-1,
                )
                neg_idx = pos_idx + 1

                prob_both_valid = pos_selection_probs[:, pos_idx:pos_idx+1].unsqueeze(1) * neg_selection_probs[:, neg_idx:neg_idx+1].unsqueeze(1)
                global_selected = 1 - prob_both_valid

                if training and self.global_ood_prob:
                    mask = torch.bernoulli(
                        torch.ones(global_selected.shape[0], 1, 1) * self.global_ood_prob,
                    ).to(global_selected.device)
                    global_selected = torch.where(
                        mask == 1,
                        torch.ones_like(global_selected),
                        global_selected,
                    )
                elif training:
                    forced_on = self.ood_dropout(torch.ones_like(global_selected))
                    global_selected = torch.ones_like(global_selected) * forced_on + (1 - forced_on) * global_selected

                pos_global_selected = global_selected
                neg_global_selected = global_selected


                if self.certificate_loss_weight:
                    pos_dynamic_selections.append(
                        pos_selection_outputs.unsqueeze(1)
                    )
                    neg_dynamic_selections.append(
                        neg_selection_outputs.unsqueeze(1)
                    )

                if self.pooling_mode in ['individual_scores', 'individual_scores_shared']:
                    combined_global_selected.append(global_selected)

            elif self.selection_mode == 'max_class_confidence':
                assert self.pooling_mode in ['individual_scores', 'individual_scores_shared'], self.pooling_mode
                dynamic_logits = self.dynamic_score_models[concept_idx](
                    c_sem[:, concept_idx:concept_idx+1] * dynamic_contexts[:, concept_idx, :self.emb_size] +
                    (1 - c_sem[:, concept_idx:concept_idx+1]) * dynamic_contexts[:, concept_idx, self.emb_size:]
                )
                global_logits = self.global_score_models[concept_idx](
                    c_sem[:, concept_idx:concept_idx+1] * global_contexts[:, concept_idx, :self.emb_size] +
                    (1 - c_sem[:, concept_idx:concept_idx+1]) * global_contexts[:, concept_idx, self.emb_size:]
                )
                global_selected = torch.softmax(
                    self.temperature * torch.concat(
                        [
                            self._max(torch.abs(dynamic_logits/self._get_dynamic_temps(concept_idx))),
                            self._max(torch.abs(global_logits/self._get_global_temps(concept_idx))),
                        ],
                        dim=-1,
                    ),
                    dim=-1,
                )[:, 1:].unsqueeze(-1)
                print("self._get_dynamic_temps(concept_idx) =", self._get_dynamic_temps(concept_idx))
                print("self._get_global_temps(concept_idx) =", self._get_global_temps(concept_idx))
                # print("torch.max(global_selected) =", torch.max(global_selected).detach().cpu().numpy())
                # print("torch.min(global_selected) =", torch.min(global_selected).detach().cpu().numpy())

                if training and self.global_ood_prob:
                    mask = torch.bernoulli(
                        torch.ones(global_selected.shape[0], 1, 1) * self.global_ood_prob,
                    ).to(global_selected.device)
                    global_selected = torch.where(
                        mask == 1,
                        torch.ones_like(global_selected),
                        global_selected,
                    )
                elif training:
                    forced_on = self.ood_dropout(torch.ones_like(global_selected))
                    global_selected = torch.ones_like(global_selected) * forced_on + (1 - forced_on) * global_selected
                pos_global_selected = global_selected
                neg_global_selected = global_selected
                if self.pooling_mode in ['individual_scores', 'individual_scores_shared']:
                    combined_global_selected.append(global_selected)

            elif self.selection_mode == 'max_concept_confidence':
                dynamic_logits = c_logits_dynamic[concept_idx]
                global_logits = c_logits_global[concept_idx]
                global_selected = torch.softmax(
                    self.temperature * torch.concat(
                        [
                            torch.abs(dynamic_logits),
                            torch.abs(global_logits),
                        ],
                        dim=-1,
                    ),
                    dim=-1,
                )[:, 1:].unsqueeze(-1)

                if training and self.global_ood_prob:
                    mask = torch.bernoulli(
                        torch.ones(pos_global_selected.shape[0], 1, 1) * self.global_ood_prob,
                    ).to(pos_global_selected.device)
                    global_selected = torch.where(
                        mask == 1,
                        torch.ones_like(global_selected),
                        global_selected,
                    )

                if training and self.global_ood_prob:
                    mask = torch.bernoulli(
                        torch.ones(global_selected.shape[0], 1, 1) * self.global_ood_prob,
                    ).to(global_selected.device)
                    global_selected = torch.where(
                        mask == 1,
                        torch.ones_like(global_selected),
                        global_selected,
                    )
                elif training:
                    forced_on = self.ood_dropout(torch.ones_like(global_selected))
                    global_selected = torch.ones_like(global_selected) * forced_on + (1 - forced_on) * global_selected

                pos_global_selected = global_selected
                neg_global_selected = global_selected
                if self.pooling_mode in ['individual_scores', 'individual_scores_shared']:
                    combined_global_selected.append(global_selected)

            elif self.selection_mode.startswith('fixed_'):
                val = float(self.selection_mode[len("fixed_"):])
                global_selected = val * torch.ones(dynamic_contexts.shape[0], 1, 1).to(dynamic_contexts.device)
                pos_global_selected = global_selected
                neg_global_selected = global_selected
                if training and self.global_ood_prob:
                    mask = torch.bernoulli(
                        torch.ones(global_selected.shape[0], 1, 1) * self.global_ood_prob,
                    ).to(global_selected.device)
                    global_selected = torch.where(
                        mask == 1,
                        torch.ones_like(global_selected),
                        global_selected,
                    )
                elif training:
                    forced_on = self.ood_dropout(torch.ones_like(global_selected))
                    global_selected = torch.ones_like(global_selected) * forced_on + (1 - forced_on) * global_selected

                if self.pooling_mode in ['individual_scores', 'individual_scores_shared']:
                    combined_global_selected.append(global_selected)
            else:
                raise ValueError(
                    f'Unsupported selection mode "{self.selection_mode}"'
                )
            if self.selection_sample:
                pos_categorical_distr = torch.concat(
                    [
                        1 - pos_global_selected[:, 0, :],
                        pos_global_selected[:, 0, :],
                    ],
                    dim=-1,
                )
                pos_global_selected = torch.nn.functional.gumbel_softmax(
                    self.temperature * pos_categorical_distr,
                    dim=-1,
                    hard=True,
                )[:, 1:].unsqueeze(1)
                neg_categorical_distr = torch.concat(
                    [
                        1 - neg_global_selected[:, 0, :],
                        neg_global_selected[:, 0, :],
                    ],
                    dim=-1,
                )
                neg_global_selected = torch.nn.functional.gumbel_softmax(
                    self.temperature * neg_categorical_distr,
                    dim=-1,
                    hard=True,
                )[:, 1:].unsqueeze(1)

            if (self.print_counter % 20 == 0) and (not training): # and (concept_idx in [0, 5, 10, 15, 20]) and (self.print_counter % 10 == 0):
            #     # if concept_idx in [0, 10, 20] and (self.print_counter % 25 == 0):
                # if (not training): # and np.mean((pos_global_selected > 0.2).detach().cpu().numpy()) > 0:
                    print()
                    print(f"\t(concept {concept_idx}) Total positive global embeddings selected for {np.mean((pos_global_selected > (self.hard_eval_selection or 0.5)).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(pos_global_selected[:]).detach().cpu().numpy(), "and a min value of", torch.min(pos_global_selected[:]).detach().cpu().numpy())
                    print(f"\t(concept {concept_idx}) Total negative global embeddings selected for {np.mean((neg_global_selected > (self.hard_eval_selection or 0.5)).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(neg_global_selected[:]).detach().cpu().numpy(), "and a min value of", torch.min(neg_global_selected[:]).detach().cpu().numpy())
                    if self.learnable_temps and concept_idx == 0: print("\t\tmax(exp(self.global_logit_temperatures)) =", torch.max(torch.exp(self.global_logit_temperatures), -1)[0].detach().cpu().numpy())
                    if self.learnable_temps and concept_idx == 0: print("\t\tmax(exp(self.dynamic_logit_temperatures)) =", torch.max(torch.exp(self.dynamic_logit_temperatures), -1)[0].detach().cpu().numpy())


            if (self.hard_eval_selection is not None) and (not training):
                pos_global_selected = torch.where(
                    pos_global_selected >= self.hard_eval_selection if self.hard_eval_selection >= 0 else pos_global_selected < self.hard_eval_selection,
                    torch.ones_like(pos_global_selected),
                    torch.zeros_like(pos_global_selected),
                )
                neg_global_selected = torch.where(
                    neg_global_selected >= self.hard_eval_selection if self.hard_eval_selection >= 0 else neg_global_selected < self.hard_eval_selection,
                    torch.ones_like(neg_global_selected),
                    torch.zeros_like(neg_global_selected),
                )
                if self.eval_majority_vote:
                    og_type = pos_global_selected.type()
                    pos_votes = pos_global_selected.sum(-1) >= pos_global_selected.shape[-1]//2
                    neg_votes = neg_global_selected.sum(-1) >= neg_global_selected.shape[-1]//2
                    total_votes = torch.logical_or(pos_votes, neg_votes)
                    pos_global_selected = total_votes.unsqueeze(-1).type(og_type)
                    neg_global_selected = total_votes.unsqueeze(-1).type(og_type)

            if self.hard_selection_value is not None:
                # This is mostly use for exploration/debugging/analysis
                pos_global_selected = torch.ones_like(pos_global_selected) * self.hard_selection_value
                neg_global_selected = torch.ones_like(pos_global_selected) * self.hard_selection_value



            if self.pooling_mode == 'concat':
                pos_embeddings.append(
                    pos_global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
                    (1 - pos_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size]
                )
                neg_embeddings.append(
                    neg_global_selected * global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
                    (1 - neg_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:]
                )
            elif self.pooling_mode == 'sigmoidal_additive':
                pos_embeddings.append(
                    global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
                    (1 - pos_global_selected) * self.sig(dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size])
                )
                neg_embeddings.append(
                    global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
                    (1 - neg_global_selected) * self.sig(dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:])
                )
            elif self.pooling_mode == 'additive':
                pos_embeddings.append(
                    global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
                    (1 - pos_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size]
                )
                neg_embeddings.append(
                    global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
                    (1 - neg_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:]
                )
            elif self.pooling_mode == 'combined':
                pos_embeddings.append(torch.concat(
                    [
                        pos_global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                        (1 - pos_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                    ],
                    dim=-1,
                ))
                neg_embeddings.append(torch.concat(
                    [
                        neg_global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                        (1 - neg_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
                    ],
                    dim=-1,
                ))
            elif self.pooling_mode in ['individual_scores', 'individual_scores_shared']:
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

        self.print_counter += 1
        pos_embeddings = torch.concat(pos_embeddings, dim=1)
        neg_embeddings = torch.concat(neg_embeddings, dim=1)
        if self.certificate_loss_weight:
            if self.selection_mode == 'global':
                self._dynamic_selections = torch.concat(
                    dynamic_selections,
                    dim=1,
                )
            elif self.selection_mode in ['individual', 'individual_joint']:
                self._pos_dynamic_selections = torch.concat(
                    pos_dynamic_selections,
                    dim=1,
                )
                self._neg_dynamic_selections = torch.concat(
                    neg_dynamic_selections,
                    dim=1,
                )


        if combined_global_selected and (
            self.pooling_mode in ['individual_scores', 'individual_scores_shared']
        ):
            self._combined_global_selected = torch.concat(combined_global_selected, dim=1)


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
        if self.certificate_loss_weight:
            for concept_idx in range(self.n_concepts):
                if self.selection_mode == 'global' and (self._dynamic_selections is not None):
                    loss += self.certificate_loss_weight * self.certificate_loss(
                        self._dynamic_selections[:, concept_idx, :],
                        torch.tensor([concept_idx for _ in range(self._dynamic_selections.shape[0])]).to(self._dynamic_selections.device)
                    )
                elif self.selection_mode in ['individual', 'individual_joint']  and (self._pos_dynamic_selections is not None):
                    # self._pos_dynamic_selections = self._pos_dynamic_selections.detach()
                    # self._neg_dynamic_selections = self._neg_dynamic_selections.detach()
                    loss += self.certificate_loss_weight * self.certificate_loss(
                        self._pos_dynamic_selections[:, concept_idx, :],
                        torch.tensor([2*concept_idx for _ in range(self._pos_dynamic_selections.shape[0])]).to(self._pos_dynamic_selections.device)
                    )

                    loss += self.certificate_loss_weight * self.certificate_loss(
                        self._neg_dynamic_selections[:, concept_idx, :],
                        torch.tensor([(2*concept_idx + 1) for _ in range(self._neg_dynamic_selections.shape[0])]).to(self._neg_dynamic_selections.device)
                    )
            self._dynamic_contexts = None
            self._variant_distrs = None
        if self.contrastive_reg and (self._c_logits_global is not None):
            loss += self.contrastive_reg * self.loss_concept(
                self.sig(torch.concat(self._c_logits_global, dim=1)),
                c,
            )
            self._c_logits_global = None

        if self.training and self.learnable_temps and self.selection_mode == 'max_class_confidence' and (
            self.global_temp_reg
        ):
            loss += self.global_temp_reg * torch.mean(
                torch.exp(self.global_logit_temperatures)
            )
            loss -= self.global_temp_reg * torch.mean(
                torch.exp(self.dynamic_logit_temperatures)
            )

        return loss



