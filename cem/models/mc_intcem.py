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


class MonteCarloIntCEM(IntAwareConceptEmbeddingModel):
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

        initial_concept_embeddings=None,
        fixed_embeddings=False,
        ood_dropout_prob=0,
        pooling_mode='concat',
        learnable_temps=False,
        mixed_probs_coeff=0.5,
        anneal_rate=1,
        min_rate=0,
        all_intervened_loss_weight=0,
        dynamic_confidence_scaling=False,
        use_only_mean_probs=True,
        calibrate_concept_probs=False,
        dyn_prob_with_global=True,
        temperature=1,
        scale_fn='original',

        # Monte carlo stuff
        deterministic=False,
        montecarlo_train_tries=20,
        montecarlo_test_tries=20,
        output_uncertainty=False,
        hard_selection_value=None,
        inference_threshold=True,

        # For debugging purposes only
        counter_limit=0,
        print_eval_only=True,
    ):
        self.print_eval_only = print_eval_only
        self.counter_limit = counter_limit
        self.print_counter = 0
        self.scale_fn = scale_fn
        self.dynamic_confidence_scaling = dynamic_confidence_scaling
        self.dyn_prob_with_global = dyn_prob_with_global
        self.temperature = temperature
        if self.dynamic_confidence_scaling:
            self._context_scale_factors = None
        self.calibrate_concept_probs = calibrate_concept_probs
        self.use_only_mean_probs = use_only_mean_probs
        self.all_intervened_loss_weight = all_intervened_loss_weight
        self.inference_threshold = inference_threshold
        self.hard_selection_value = hard_selection_value
        self._construct_c2y_model = False
        self.mixed_probs_coeff = mixed_probs_coeff
        self.anneal_rate = anneal_rate
        self.min_rate = min_rate
        bottleneck_size = 2 * emb_size * n_concepts if pooling_mode == 'combined' else emb_size * n_concepts
        self.montecarlo_train_tries = montecarlo_train_tries
        self.montecarlo_test_tries = montecarlo_test_tries
        self.deterministic = deterministic
        self.ood_dropout_prob = ood_dropout_prob
        self.output_uncertainty = output_uncertainty
        self.learnable_temps = learnable_temps

        self._mixed_stds = None

        super(MonteCarloIntCEM, self).__init__(
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

        else:
            raise ValueError(
                f'Unsupported pooling mode "{self.pooling_mode}"'
            )

        self.ood_dropout_prob = ood_dropout_prob
        self.global_concept_context_generators = torch.nn.Linear(
            list(
                self.pre_concept_model.modules()
            )[-1].out_features,
            self.emb_size,
        )
        self.concept_scales = torch.nn.Parameter(
            torch.rand(n_concepts),
            requires_grad=True,
        )
        self.ood_uncertainty_thresh = torch.nn.Parameter(
            torch.zeros(1),
            requires_grad=False,
        )
        if self.calibrate_concept_probs:
            self.concept_platt_scales = torch.nn.Parameter(
                torch.ones(n_concepts),
                requires_grad=False,
            )
            self.concept_platt_biases = torch.nn.Parameter(
                torch.zeros(n_concepts),
                requires_grad=False,
            )
        self.logit_temperatures = torch.nn.Parameter(
            torch.ones(n_concepts, n_tasks),
            requires_grad=learnable_temps,
        )

    def _uncertainty_based_context_addition(self, concept_probs, temperature=1):
        if self.dynamic_confidence_scaling:
            # Then we only select to add a context when the uncertainty is far
            # from the extremes
            if self.scale_fn == 'original':
                # The happy typo/mistake function that almost acts as a linear
                # function in [0, 1] when the temperature is 1
                return 1 - 4 * (
                    torch.sigmoid(temperature * (concept_probs - 0.5)) *
                    torch.sigmoid(-temperature * (0.5 - concept_probs))
                )
            elif self.scale_fn == 'sigmoidal_mix':
                # The originally intended thresholding function
                return torch.relu(1 - 4 * (
                    torch.sigmoid(temperature * (concept_probs - 0.5)) *
                    torch.sigmoid(-temperature * (concept_probs - 0.5))
                ))
            elif self.scale_fn == 'tanh':
                # A sharper version of the signed mixing
                return torch.tanh(self.temperature * (concept_probs - 0.5))
            elif self.scale_fn == 'linear':
                return self.temperature * (concept_probs - 0.5)
            elif self.scale_fn in ['entropy', 'entr']:
                entr = (
                    -concept_probs * torch.log2(concept_probs + 1e-6) -
                    (1 - concept_probs) * torch.log2(1 - concept_probs + 1e-6)
                )
                return self.temperature * (1 - entr)
            elif self.scale_fn in ['mean_entropy', 'mean_entr']:
                entr = (
                    -concept_probs * torch.log2(concept_probs + 1e-6) -
                    (1 - concept_probs) * torch.log2(1 - concept_probs + 1e-6)
                )
                scaled = self.temperature * (1 - entr)
                reduced = torch.mean(scaled, dim=-1, keepdim=True)
                return reduced.expand(-1, scaled.shape[-1])
            elif self.scale_fn in ['max_entropy', 'max_entr']:
                entr = (
                    -concept_probs * torch.log2(concept_probs + 1e-6) -
                    (1 - concept_probs) * torch.log2(1 - concept_probs + 1e-6)
                )
                entr = torch.max(entr, dim=-1, keepdim=True)[0].expand(-1, entr.shape[-1])
                return self.temperature * (1 - entr)
            elif self.scale_fn in ['diff_max_entropy', 'diff_max_entr']:
                entr = (
                    -concept_probs * torch.log2(concept_probs + 1e-6) -
                    (1 - concept_probs) * torch.log2(1 - concept_probs + 1e-6)
                )
                entr = torch.sum(
                    entr * torch.log(torch.softmax(10 * entr, dim=-1)),
                    dim=-1,
                    keepdim=True,
                ).expand(-1, entr.shape[-1])
                return self.temperature * (1 - entr)
            elif self.scale_fn in ['exponential_entropy', 'exp_entr']:
                eps = 1e-4
                numerator = torch.pow(concept_probs, self.temperature)
                denominator = numerator + torch.pow(
                    (1 - concept_probs),
                    self.temperature,
                )
                adj_prob = numerator / (denominator + eps)
                return 1 - (
                    -adj_prob * torch.log2(adj_prob + eps) -
                    (1 - adj_prob) * torch.log2(1 - adj_prob + eps)
                )
            elif self.scale_fn in ['exponential', 'exp']:
                return torch.pow(
                    2.0 * (concept_probs - 0.5),
                    2 * self.temperature,
                )
            elif self.scale_fn in ['abs', 'absolute']:
                return self.temperature * torch.abs(concept_probs - 0.5)
            else:
                raise ValueError(
                    f'Unsupported scale function "{self.scale_fn}"!'
                )
        else:
            # Else we otherwise always select the dynamic context
            return torch.ones_like(concept_probs)


    def _predict_labels(self, bottleneck, **task_loss_kwargs):
        outputs = []
        if bottleneck.shape[-1] == 2:
            # Then no test time sampling was done!! So let's just use the
            # normal mixed bottleneck. This will always be the first
            # trial output
            outputs.append(
                self.logit_temperatures[0, 0] *
                self.c2y_model(bottleneck[:, :, :, 0])
            )
        else:
            for trial_idx in range(2, bottleneck.shape[-1]):
                out_vals = self.c2y_model(bottleneck[:, :, :, trial_idx])
                out_vals = self.logit_temperatures[0, 0] * out_vals
                outputs.append(out_vals.unsqueeze(-1))
            outputs = torch.concat(outputs, dim=-1)
        self._mixed_stds = torch.std(outputs, dim=-1)
        outputs = torch.mean(outputs, dim=-1)
        if (not self.training) and self.inference_threshold and (
            self.ood_dropout_prob > 0
        ) and (not self.dynamic_confidence_scaling):
            best_outputs = torch.argmax(outputs, dim=-1).unsqueeze(-1)
            best_confs = torch.FloatTensor(
                [self._mixed_stds[x, best_outputs[x]] for x in range(self._mixed_stds.shape[0])]
            ).to(self._mixed_stds.device)
            context_selection = torch.where(
                best_confs <= self.ood_uncertainty_thresh,
                torch.zeros(self._mixed_stds.shape[0]).to(self._mixed_stds.device),
                torch.ones(self._mixed_stds.shape[0]).to(self._mixed_stds.device),
            ).unsqueeze(-1).unsqueeze(-1)
            new_bottleneck = context_selection * bottleneck[:, :, :, 0] + (
                (1 - context_selection) * bottleneck[:, :, :, 1]
            )
            return self.logit_temperatures[0, 0] * self.c2y_model(new_bottleneck)
        return outputs

    def _construct_c2y_input(
        self,
        pos_embeddings,
        neg_embeddings,
        probs,
        **task_loss_kwargs,
    ):
        # We will generate several versions of the bottleneck with different
        # masks.Then, downstream in the line with _predict_labels, we will
        # unpack them, make a label prediction, and compute the mean and
        # variance of all samples
        extra_scale = 1
        if self.deterministic or (self.hard_selection_value is not None):
            n_trials = 1
        elif not self.training:
            if self.montecarlo_test_tries == 0:
                # Then we will interpret this as a normal dropout rescaling
                # during inference
                n_trials = 1
                extra_scale = (
                    self.ood_dropout_prob if self.ood_dropout_prob > 0
                    else 1
                )
            else:
                n_trials = max(self.montecarlo_test_tries, 0)
        else:
            n_trials = max(self.montecarlo_train_tries, 0)


        global_pos_embeddings = pos_embeddings[:, :, :self.emb_size]
        contextual_pos_embeddings = pos_embeddings[:, :, self.emb_size:]
        if self.dynamic_confidence_scaling:
            contextual_pos_embeddings = (
                contextual_pos_embeddings *
                self._context_scale_factors.unsqueeze(-1)
            )
        global_neg_embeddings = neg_embeddings[:, :, :self.emb_size]
        contextual_neg_embeddings = neg_embeddings[:, :, self.emb_size:]
        if self.dynamic_confidence_scaling:
            contextual_neg_embeddings = (
                contextual_neg_embeddings *
                self._context_scale_factors.unsqueeze(-1)
            )
        bottlenecks = []
        # The first two elements of the array will always be the contextual
        # mixed embedding followed by just the one using the global embeddings
        combined_pos_embs = global_pos_embeddings + \
            extra_scale * contextual_pos_embeddings
        combined_neg_embs = global_neg_embeddings + \
            extra_scale * contextual_neg_embeddings
        new_bottleneck = (
            combined_pos_embs * torch.unsqueeze(probs, dim=-1) + (
                combined_neg_embs * (
                    1 - torch.unsqueeze(probs, dim=-1)
                )
            )
        )
        bottlenecks.append(new_bottleneck.unsqueeze(-1))
        new_bottleneck = (
            global_pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                global_neg_embeddings * (
                    1 - torch.unsqueeze(probs, dim=-1)
                )
            )
        )
        bottlenecks.append(new_bottleneck.unsqueeze(-1))
        for _ in range(n_trials):
            if self.hard_selection_value is not None:
                context_selected = (1 - self.hard_selection_value) * torch.ones(
                    global_pos_embeddings.shape[0], self.n_concepts, 1
                ).to(global_pos_embeddings.device)
            elif self.deterministic:
                context_selected = torch.ones(
                    global_pos_embeddings.shape[0], self.n_concepts, 1
                ).to(global_pos_embeddings.device)
            else:
                # Then we generate a simple random mask
                dropout_prob = self.ood_dropout_prob
                if self.anneal_rate != 1:
                    # Then time to do some annealing
                    if self.anneal_rate > 0:
                        dropout_prob = max(
                            self.min_rate,
                            np.exp(
                                -self.anneal_rate * dropout_prob *
                                self.current_steps.detach().cpu().numpy()[0]
                            ),
                        )
                    else:
                        dropout_prob = 1 - max(
                            self.min_rate,
                            np.exp(
                                self.anneal_rate * (1 - dropout_prob) *
                                self.current_steps.detach().cpu().numpy()[0]
                            ),
                        )
                    dropout_prob = float(np.clip(dropout_prob, 0, 1))
                # if self.current_steps.detach() % 10:
                #     print("For step", self.current_steps.detach(), "dropout rate is", dropout_prob)
                context_selected = torch.bernoulli(
                    torch.ones(global_pos_embeddings.shape[0], self.n_concepts, 1) * (1 - dropout_prob)
                ).to(global_pos_embeddings.device)
            combined_pos_embs = global_pos_embeddings + (
                context_selected * contextual_pos_embeddings
            )
            combined_neg_embs = global_neg_embeddings + (
                context_selected * contextual_neg_embeddings
            )
            new_bottleneck = (
                combined_pos_embs * torch.unsqueeze(probs, dim=-1) + (
                    combined_neg_embs * (
                        1 - torch.unsqueeze(probs, dim=-1)
                    )
                )
            )
            bottlenecks.append(new_bottleneck.unsqueeze(-1))
        return torch.concat(bottlenecks, dim=-1)

    def _construct_rank_model_input(self, bottleneck, prev_interventions):
        bottleneck = bottleneck[:, :, :, 0] # We always use the dynamic + global embeddings
        cat_inputs = [
            bottleneck.reshape(bottleneck.shape[0], -1),
            prev_interventions,
        ]
        return torch.concat(
            cat_inputs,
            dim=-1,
        )

    def _new_tail_results(
        self,
        x=None,
        c=None,
        y=None,
        c_sem=None,
        bottleneck=None,
        y_pred=None,
    ):
        tail_results = []
        if (
            (self._mixed_stds is not None) and
            (self.output_uncertainty)
        ):
            tail_results.append(self._mixed_stds)
            self._mixed_stds = None
        return tail_results

    def _generate_dynamic_concept(self, pre_c, concept_idx):
        context = self.concept_context_generators[concept_idx](pre_c)
        return context


    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        extra_outputs = {}
        if latent is None:
            pre_c = self.pre_concept_model(x)
            global_emb_center = self.global_concept_context_generators(pre_c)
            dynamic_contexts = []
            for concept_idx in range(self.n_concepts):
                dynamic_context = self._generate_dynamic_concept(pre_c, concept_idx=concept_idx)
                dynamic_contexts.append(torch.unsqueeze(dynamic_context, dim=1))
            dynamic_contexts = torch.cat(dynamic_contexts, axis=1)
            self._dynamic_context = dynamic_contexts

            global_context_pos = self.concept_embeddings[:, 0, :].unsqueeze(0).expand(pre_c.shape[0], -1, -1)
            global_context_neg = self.concept_embeddings[:, 1, :].unsqueeze(0).expand(pre_c.shape[0], -1, -1)
            global_contexts = torch.concat(
                [global_context_pos, global_context_neg],
                dim=-1,
            )
            latent = dynamic_contexts, global_contexts
        else:
            dynamic_contexts, global_contexts = latent

        # Now we can compute all the probabilites!
        c_sem = []

        for concept_idx in range(self.n_concepts):
            if self.shared_prob_gen:
                prob_gen = self.concept_prob_generators[0]
            else:
                prob_gen = self.concept_prob_generators[concept_idx]
            if self.dyn_prob_with_global:
                if self.hard_selection_value is None:
                    dyn_logits = prob_gen(
                        global_contexts[:, concept_idx, :] +
                        dynamic_contexts[:, concept_idx, :]
                    )
                else:
                    # Else we adjust the contextual embeddings here directly
                    dyn_logits = prob_gen(
                        global_contexts[:, concept_idx, :] +
                        (1 - self.hard_selection_value) * dynamic_contexts[:, concept_idx, :]
                    )
            else:
                dyn_logits = prob_gen(
                    dynamic_contexts[:, concept_idx, :]
                )
            dyn_prob = self.sig(
                self._concept_platt_scaling(dyn_logits, concept_idx)
            )
            mixed_factor = self.mixed_probs_coeff
            if self.anneal_rate != 1:
                # Then time to do some annealing
                if self.anneal_rate > 0 :
                    mixed_factor = 1 - max(
                        self.min_rate,
                        np.exp(
                            -self.anneal_rate * (1 - mixed_factor) *
                            self.current_steps.detach().cpu().numpy()[0]
                        )
                    )
                else:
                    mixed_factor = max(
                        self.min_rate,
                        np.exp(
                            self.anneal_rate * mixed_factor *
                            self.current_steps.detach().cpu().numpy()[0]
                        )
                    )
                mixed_factor = float(np.clip(mixed_factor, 0, 1))

            if mixed_factor != 1:
                pos_anchor = self.concept_embeddings[concept_idx:concept_idx+1, 0, :].expand(global_emb_center.shape[0], -1)
                neg_anchor = self.concept_embeddings[concept_idx:concept_idx+1, 1, :].expand(global_emb_center.shape[0], -1)
                neg_dist = (
                    neg_anchor - global_emb_center
                ).pow(2).sum(-1).sqrt()
                pos_dist = (
                    pos_anchor - global_emb_center
                ).pow(2).sum(-1).sqrt()
                global_logits = (
                    torch.exp(self.concept_scales[concept_idx]) *
                    (neg_dist - pos_dist).unsqueeze(-1)
                )
                global_prob = self.sig(
                    self._concept_platt_scaling(global_logits, concept_idx)
                )
                prob = (
                    mixed_factor * dyn_prob +
                    (1 -mixed_factor) * global_prob
                )
            else:
                prob = dyn_prob
            c_sem.append(prob)

        c_sem = torch.cat(c_sem, axis=-1)
        self._context_scale_factors = self._uncertainty_based_context_addition(
            concept_probs=c_sem,
            temperature=self.temperature,
        )
        if self.hard_selection_value is not None:
            self._context_scale_factors = 1 - self.hard_selection_value * torch.ones_like(
                self._context_scale_factors
            )
        if self.counter_limit and (self.print_counter % self.counter_limit == 0) and ((not training) or (not self.print_eval_only)):
            x = self._context_scale_factors.detach().cpu().numpy()
            print("For step", self.current_steps.detach(), "average mixing factor is", np.mean(x))
            print("\tMax mixing factor is", np.max(x))
            print("\tMin mixing factor is", np.min(x))
            print("\tMean mixing factor is", np.mean(x))
            print("\tMean prob is", np.mean(c_sem.detach().cpu().numpy()))
            print("\tMax prob is", np.max(c_sem.detach().cpu().numpy()))
            print("\tMin prob is", np.min(c_sem.detach().cpu().numpy()))
        self.print_counter += 1


        pos_embeddings = torch.concat(
            [
                global_contexts[:, :, :self.emb_size],
                dynamic_contexts[:, :, :self.emb_size],
            ],
            dim=-1
        )
        neg_embeddings = torch.concat(
            [
                global_contexts[:, :, self.emb_size:],
                dynamic_contexts[:, :, self.emb_size:],
            ],
            dim=-1
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
        if self.all_intervened_loss_weight != 0:
            global_context_pos = self.concept_embeddings[:, 0, :].unsqueeze(0).expand(c.shape[0], -1, -1)
            global_context_neg = self.concept_embeddings[:, 1, :].unsqueeze(0).expand(c.shape[0], -1, -1)
            new_bottleneck = (
                global_context_pos * torch.unsqueeze(c, dim=-1) + (
                    global_context_neg * (
                        1 - torch.unsqueeze(c, dim=-1)
                    )
                )
            )
            new_y_logits = self.logit_temperatures[0, 0] * self.c2y_model(new_bottleneck)
            loss += self.all_intervened_loss_weight * self.loss_task(
                new_y_logits if new_y_logits.shape[-1] > 1 else new_y_logits.reshape(-1),
                y,
            )
        return loss


    def _concept_platt_scaling(self, logits, concept_idx):
        if self.calibrate_concept_probs:
            return (
                self.concept_platt_scales[concept_idx] * logits +
                self.concept_platt_biases[concept_idx]
            )
        return logits

    def freeze_global_components(self):
        self.concept_embeddings.requires_grad = False
        self.concept_scales.requires_grad = False
        for param in self.global_concept_context_generators.parameters():
            param.requires_grad = False
        for param in self.pre_concept_model.parameters():
            param.requires_grad = False

    def freeze_ood_separator(self):
        pass

    def unfreeze_ood_separator(self):
        pass

    def unfreeze_calibration_components(
        self,
        unfreeze_dynamic=True,
        unfreeze_global=True,
    ):
        if self.calibrate_concept_probs:
            self.concept_platt_scales.requires_grad = True
            self.concept_platt_biases.requires_grad = True
        else:
            self.logit_temperatures.requires_grad = True

    def freeze_calibration_components(
        self,
        freeze_dynamic=True,
        freeze_global=True,
    ):
        if self.calibrate_concept_probs:
            self.concept_platt_scales.requires_grad = False
            self.concept_platt_biases.requires_grad = False
        else:
            self.logit_temperatures.requires_grad = False

    def unfreeze_global_components(self):
        self.concept_embeddings.requires_grad = True
        self.concept_scales.requires_grad = True
        for param in self.global_concept_context_generators.parameters():
            param.requires_grad = True
        for param in self.pre_concept_model.parameters():
            param.requires_grad = True



