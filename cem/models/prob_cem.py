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

# class ProbConceptEmbeddingModel(IntAwareConceptEmbeddingModel):
#     def __init__(
#         self,
#         n_concepts,
#         n_tasks,
#         emb_size=16,
#         training_intervention_prob=0.25,
#         embedding_activation="leakyrelu",
#         concept_loss_weight=1,

#         c2y_model=None,
#         c2y_layers=None,
#         c_extractor_arch=utils.wrap_pretrained_model(resnet50),
#         output_latent=False,

#         optimizer="adam",
#         momentum=0.9,
#         learning_rate=0.01,
#         weight_decay=4e-05,
#         lr_scheduler_factor=0.1,
#         lr_scheduler_patience=10,
#         weight_loss=None,
#         task_class_weights=None,

#         active_intervention_values=None,
#         inactive_intervention_values=None,
#         intervention_policy=None,
#         output_interventions=False,

#         top_k_accuracy=None,

#         intervention_task_discount=1.1,
#         intervention_weight=5,
#         concept_map=None,
#         use_concept_groups=True,

#         rollout_init_steps=0,
#         int_model_layers=None,
#         int_model_use_bn=True,
#         num_rollouts=1,

#         # Parameters regarding how we select how many concepts to intervene on
#         # in the horizon of a current trajectory (this is the lenght of the
#         # trajectory)
#         max_horizon=6,
#         initial_horizon=2,
#         horizon_rate=1.005,

#         # Experimental/debugging arguments
#         intervention_discount=1,
#         include_only_last_trajectory_loss=True,
#         task_loss_weight=1,
#         intervention_task_loss_weight=1,

#         ##################################
#         # New arguments
#         #################################

#         temperature=1,
#         n_concept_variants=5,
#         initial_concept_embeddings=None,
#         fixed_embeddings=False,
#         initial_log_variances=None,
#         fixed_variances=False,
#         attention_fn='softmax',
#         ood_dropout_prob=0,
#         pooling_mode='concat',
#         selection_mode='z_score',
#         kl_loss_weight=0,
#         box_temparature=1,
#         threshold=None,
#         fallback_mode='mixed',
#     ):
#         self._construct_c2y_model = False
#         super(ProbConceptEmbeddingModel, self).__init__(
#             n_concepts=n_concepts,
#             n_tasks=n_tasks,
#             emb_size=emb_size,
#             training_intervention_prob=training_intervention_prob,
#             embedding_activation=embedding_activation,
#             concept_loss_weight=concept_loss_weight,
#             c2y_model=c2y_model,
#             c2y_layers=c2y_layers,
#             c_extractor_arch=c_extractor_arch,
#             output_latent=output_latent,
#             optimizer=optimizer,
#             momentum=momentum,
#             learning_rate=learning_rate,
#             weight_decay=weight_decay,
#             lr_scheduler_factor=lr_scheduler_factor,
#             lr_scheduler_patience=lr_scheduler_patience,
#             weight_loss=weight_loss,
#             task_class_weights=task_class_weights,
#             active_intervention_values=active_intervention_values,
#             inactive_intervention_values=inactive_intervention_values,
#             intervention_policy=intervention_policy,
#             output_interventions=output_interventions,
#             top_k_accuracy=top_k_accuracy,
#             intervention_task_discount=intervention_task_discount,
#             intervention_weight=intervention_weight,
#             concept_map=concept_map,
#             use_concept_groups=use_concept_groups,
#             rollout_init_steps=rollout_init_steps,
#             int_model_layers=int_model_layers,
#             int_model_use_bn=int_model_use_bn,
#             num_rollouts=num_rollouts,
#             max_horizon=max_horizon,
#             initial_horizon=initial_horizon,
#             horizon_rate=horizon_rate,
#             intervention_discount=intervention_discount,
#             include_only_last_trajectory_loss=include_only_last_trajectory_loss,
#             task_loss_weight=task_loss_weight,
#             intervention_task_loss_weight=intervention_task_loss_weight,
#         )
#         self.selection_mode = selection_mode
#         self.temperature = temperature
#         self.n_concept_variants = n_concept_variants
#         self.attention_fn = attention_fn
#         self.ood_dropout_prob = ood_dropout_prob
#         self.ood_dropout = torch.nn.Dropout(
#             p=(1 - ood_dropout_prob),  # We do 1 - ood_prob as this will be applied to the selection of the global embedding
#         )

#         # Let's generate the global embeddings we will use
#         # Must output logits with shape (B, n_concept_variants)
#         if (initial_concept_embeddings is False) or (
#             initial_concept_embeddings is None
#         ):
#             initial_concept_embeddings = torch.normal(
#                 torch.zeros(self.n_concepts, n_concept_variants, 2, emb_size),
#                 torch.ones(self.n_concepts, n_concept_variants, 2, emb_size),
#             )
#         else:
#             if isinstance(initial_concept_embeddings, np.ndarray):
#                 initial_concept_embeddings = torch.FloatTensor(
#                     initial_concept_embeddings
#                 )
#             emb_size = initial_concept_embeddings.shape[-1]
#         self.concept_embeddings = torch.nn.Parameter(
#             initial_concept_embeddings,
#             requires_grad=(not fixed_embeddings),
#         )

#         # Then their corresponding log variances
#         if (initial_log_variances is False) or (
#             initial_log_variances is None
#         ):
#             initial_log_variances = torch.normal(
#                 torch.zeros(self.n_concepts, n_concept_variants, 2, emb_size),
#                 torch.ones(self.n_concepts, n_concept_variants, 2, emb_size),
#             )
#         else:
#             if isinstance(initial_log_variances, np.ndarray):
#                 initial_log_variances = torch.FloatTensor(
#                     initial_log_variances
#                 )
#         self.concept_log_devs = torch.nn.Parameter(
#             initial_log_variances,
#             requires_grad=(not fixed_variances),
#         )

#         # Set up any attention model needed if we have multiple variants per
#         # concept
#         if self.n_concept_variants > 1:
#             self.attn_model = torch.nn.ModuleList([
#                 torch.nn.Sequential(
#                     torch.nn.Linear(
#                         list(
#                             self.pre_concept_model.modules()
#                         )[-1].out_features,
#                         64,
#                     ),
#                     torch.nn.LeakyReLU(),
#                     torch.nn.Linear(
#                         64,
#                         32,
#                     ),
#                     torch.nn.Linear(
#                         32,
#                         self.n_concept_variants,
#                     ),
#                 )
#                 for _ in range(self.n_concepts)
#             ])
#         else:
#             self.attn_model = [
#                 lambda x: torch.ones(x.shape[0], self.n_concept_variants).to(x.device)
#                 for _ in range(self.n_concepts)
#             ]

#         self.print_counter = 0

#         self.pooling_mode = pooling_mode
#         if self.pooling_mode == 'concat':
#             # Then nothing to do or change here
#             if c2y_model is None:
#                 # Else we construct it here directly
#                 units = [
#                     n_concepts * emb_size
#                 ] + (c2y_layers or []) + [n_tasks]
#                 layers = [torch.nn.Flatten(1, -1)]
#                 for i in range(1, len(units)):
#                     layers.append(torch.nn.Linear(units[i-1], units[i]))
#                     if i != len(units) - 1:
#                         layers.append(torch.nn.LeakyReLU())
#                 self.c2y_model = torch.nn.Sequential(*layers)
#             else:
#                 self.c2y_model = c2y_model
#             self.global_c2y_model = self.c2y_model
#         elif self.pooling_mode == 'individual_scores':
#             self.dynamic_score_models = torch.nn.ModuleList([
#                 torch.nn.Sequential(
#                     torch.nn.Linear(
#                         self.emb_size,
#                         n_tasks,
#                     ),
#                 )
#                 for _ in range(self.n_concepts)
#             ])
#             self.global_score_models = torch.nn.ModuleList([
#                 torch.nn.Sequential(
#                     torch.nn.Linear(
#                         self.emb_size,
#                         n_tasks,
#                     ),
#                 )
#                 for _ in range(self.n_concepts)
#             ])

#             self.c2y_model = self._individual_score_mix
#             self.global_c2y_model = lambda x: self._individual_score_mix(x, only_global=True)
#         else:
#             raise ValueError(
#                 f'Unsupported pooling mode "{self.pooling_mode}"'
#             )

#         self.kl_loss_weight = kl_loss_weight


#         self.box_temparature = box_temparature
#         if threshold is None:
#             self.thresholds = torch.nn.Parameter(
#                 torch.normal(
#                     torch.zeros(self.n_concepts, 2),
#                     torch.ones(self.n_concepts, 2),
#                 ),
#                 requires_grad=True,
#             )
#         else:
#             self.thresholds = threshold * torch.ones(self.n_concepts, 2)


#         self._dynamic_contexts = None
#         self._variant_distrs = None

#         self.fallback_mode = fallback_mode
#         if fallback_mode == 'global':
#             self.fallback_concept_embs = torch.nn.Parameter(
#                 torch.normal(
#                     torch.zeros(self.n_concepts, 2, self.emb_size),
#                     torch.ones(self.n_concepts, 2, self.emb_size),
#                 ),
#                 requires_grad=True,
#             )
#         elif fallback_mode == 'mixed':
#             pass
#         else:
#             raise ValueError(
#                 f'Unsupported fallback mode "{fallback_mode}"'
#             )

#     def _construct_c2y_input(
#         self,
#         pos_embeddings,
#         neg_embeddings,
#         probs,
#         **task_loss_kwargs,
#     ):
#         bottleneck = (
#             pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
#                 neg_embeddings * (
#                     1 - torch.unsqueeze(probs, dim=-1)
#                 )
#             )
#         )
#         return bottleneck

#     def _individual_score_mix(self, bottleneck, only_global=False):
#         # global_selected will have shape (B, n_concepts)
#         global_selected = self._combined_global_selected.squeeze(-1)
#         if only_global:
#             global_selected = torch.ones_like(global_selected)

#         output_logits = None
#         for concept_idx in range(self.n_concepts):
#             global_embs = bottleneck[:, concept_idx, :self.emb_size]
#             global_logits = self.global_score_models[concept_idx](global_embs)

#             dynamic_embs = bottleneck[:, concept_idx, self.emb_size:]
#             dynamic_logits = self.dynamic_score_models[concept_idx](dynamic_embs)

#             combined_scores = (
#                 global_selected[:, concept_idx:concept_idx+1] * global_logits +
#                 (1 - global_selected[:, concept_idx:concept_idx+1]) * dynamic_logits
#             )

#             if output_logits is None:
#                 output_logits = combined_scores
#             else:
#                 output_logits = output_logits + combined_scores

#         return output_logits


#     def _construct_rank_model_input(self, bottleneck, prev_interventions):
#         if self.pooling_mode == 'individual_scores':
#             bottleneck = bottleneck[:, :, self.emb_size:]
#         cat_inputs = [
#             bottleneck.reshape(bottleneck.shape[0], -1),
#             prev_interventions,
#         ]
#         return torch.concat(
#             cat_inputs,
#             dim=-1,
#         )

#     def _box_function(self, x, concept_idx=None, left_bound=None, right_bound=None):
#         if left_bound is None:
#             assert concept_idx is not None
#             left_bound = 0.5*torch.sigmoid(self.left_bounds[concept_idx])
#         left_barrier = torch.tanh(self.box_temperature * (x - left_bound))
#         if right_bound is None:
#             assert concept_idx is not None
#             right_bound = 0.5 + 0.5*torch.sigmoid(self.right_bounds[concept_idx])
#         right_barrier = torch.tanh(self.box_temperature * (x - right_bound))
#         return 0.5 * (left_barrier - right_barrier)


#     def _left_box_function(self, x, concept_idx, left_bound):
#         return torch.sigmoid(self.box_temperature * (x - left_bound))

#     def _right_box_function(self, x, concept_idx, right_bound):
#         return torch.sigmoid(self.box_temperature * (right_bound - x))

#     def _fuzzy_or(self, a, b, sharpness=5):
#         # return torch.log(torch.exp(sharpness * a) + torch.exp(sharpness * b))/sharpness
#         return a + b - a * b

#     def _generate_dynamic_concept(self, pre_c, concept_idx):
#         context = self.concept_context_generators[concept_idx](pre_c)
#         return context

#     def _generate_concept_embeddings(
#         self,
#         x,
#         latent=None,
#         training=False,
#     ):
#         extra_outputs = {}
#         if latent is None:
#             pre_c = self.pre_concept_model(x)
#             dynamic_contexts = []
#             global_contexts = []
#             variant_distrs = []

#             # First predict all the concept probabilities
#             for concept_idx in range(self.n_concepts):
#                 if self.shared_prob_gen:
#                     prob_gen = self.concept_prob_generators[0]
#                 else:
#                     prob_gen = self.concept_prob_generators[concept_idx]
#                 dynamic_context = self._generate_dynamic_concept(pre_c, concept_idx=concept_idx)
#                 dynamic_contexts.append(torch.unsqueeze(dynamic_context, dim=1))

#                 # distr has shape (B, n_concept_variants)
#                 if self.attention_fn == 'sigmoid':
#                     distr = torch.sigmoid(
#                         self.temperature * self.attn_model[concept_idx](pre_c)
#                     )

#                 elif self.attention_fn == 'softmax':
#                     distr = torch.softmax(
#                         self.temperature * self.attn_model[concept_idx](pre_c),
#                         dim=-1,
#                     )

#                 elif self.attention_fn == 'soft_gumbel':
#                     distr = torch.nn.functional.gumbel_softmax(
#                         self.temperature * self.attn_model[concept_idx](pre_c),
#                         dim=-1,
#                         hard=False,
#                     )

#                 elif self.attention_fn == 'hard_gumbel':

#                     distr = torch.nn.functional.gumbel_softmax(
#                         self.temperature * self.attn_model[concept_idx](pre_c),
#                         dim=-1,
#                         hard=True,
#                     )
#                 else:
#                     raise ValueError(
#                         f'Unrecognized attention_fn "{self.attention_fn}".'
#                     )
#                 if self.fallback_mode == 'mixed':
#                     #  global_context_pos has shape (1, n_concept_variants, emb_size)
#                     global_context_pos = self.concept_embeddings[concept_idx:concept_idx+1, :, 0, :]
#                     #  global_context_pos has shape (B, n_concept_variants, emb_size)
#                     global_context_pos = global_context_pos.expand(pre_c.shape[0], -1, -1)
#                     #  global_context_pos now has shape (B, emb_size)
#                     global_context_pos = (
#                         distr.unsqueeze(-1) * global_context_pos
#                     ).sum(1)

#                     #  global_context_neg has shape (1, n_concept_variants, emb_size)
#                     global_context_neg = self.concept_embeddings[concept_idx:concept_idx+1, :, 1, :]
#                     #  global_context_neg has shape (B, n_concept_variants, emb_size)
#                     global_context_neg = global_context_neg.expand(pre_c.shape[0], -1, -1)
#                     #  global_context_neg now has shape (B, emb_size)
#                     global_context_neg = (
#                         distr.unsqueeze(-1) * global_context_neg
#                     ).sum(1)
#                 elif self.fallback_mode == 'global':
#                     global_context_pos = self.fallback_concept_embs[concept_idx:concept_idx+1, 0, :].expand(pre_c.shape[0], -1)
#                     global_context_neg = self.fallback_concept_embs[concept_idx:concept_idx+1, 1, :].expand(pre_c.shape[0], -1)
#                 else:
#                     raise ValueError(
#                         f'Unsupported fallback_mode "{self.fallback_mode}"'
#                     )


#                 global_context = torch.concat(
#                     [global_context_pos, global_context_neg],
#                     dim=-1,
#                 )
#                 global_contexts.append(torch.unsqueeze(global_context, dim=1))
#                 variant_distrs.append(distr.unsqueeze(1))
#             dynamic_contexts = torch.cat(dynamic_contexts, axis=1)
#             global_contexts = torch.cat(global_contexts, axis=1)
#             latent = dynamic_contexts, global_contexts, variant_distrs
#         else:
#             dynamic_contexts, global_contexts, variant_distrs = latent

#         self._variant_distrs = torch.concat(variant_distrs, dim=1)

#         # Now we can compute all the probabilites!
#         c_sem = []
#         c_logits = []
#         self._dynamic_contexts = dynamic_contexts

#         prob_contexts = dynamic_contexts
#         for concept_idx in range(self.n_concepts):
#             if self.shared_prob_gen:
#                 prob_gen = self.concept_prob_generators[0]
#             else:
#                 prob_gen = self.concept_prob_generators[concept_idx]
#             c_logits.append(prob_gen(prob_contexts[:, concept_idx, :]))
#             prob = self.sig(c_logits[-1])
#             c_sem.append(prob)
#         c_sem = torch.cat(c_sem, axis=-1)
#         c_logits = torch.cat(c_logits, axis=-1)

#         pos_embeddings = []
#         neg_embeddings = []
#         for concept_idx in range(self.n_concepts):
#             if self.selection_mode == 'z_score':
#                 pos_embs = dynamic_contexts[:, concept_idx, :self.emb_size]
#                 pos_global_selected = 0
#                 for var_idx in range(self.n_concept_variants):
#                     pos_mean = self.concept_embeddings[concept_idx:concept_idx+1, var_idx, 0, :].expand(dynamic_contexts.shape[0], -1)
#                     pos_std = torch.exp(
#                         self.concept_log_devs[concept_idx:concept_idx+1, var_idx, 0, :].expand(dynamic_contexts.shape[0], -1)
#                     )
#                     pos_z_scores = torch.abs(pos_embs - pos_mean) / pos_std
#                     pos_z_scores = torch.mean(pos_z_scores, dim=-1).unsqueeze(-1).unsqueeze(-1)
#                     pos_global_selected = pos_global_selected + self._variant_distrs[:, concept_idx:concept_idx+1, var_idx:var_idx+1] * torch.sigmoid(
#                         self.box_temparature * (pos_z_scores - self.thresholds[concept_idx, 0])
#                     )

#                 neg_embs = dynamic_contexts[:, concept_idx, self.emb_size:]
#                 neg_global_selected = 0
#                 for var_idx in range(self.n_concept_variants):
#                     neg_mean = self.concept_embeddings[concept_idx:concept_idx+1, var_idx, 1, :].expand(dynamic_contexts.shape[0], -1)
#                     neg_std = torch.exp(
#                         self.concept_log_devs[concept_idx:concept_idx+1, var_idx, 1, :].expand(dynamic_contexts.shape[0], -1)
#                     )
#                     neg_z_scores = torch.abs(neg_embs - neg_mean) / neg_std
#                     neg_z_scores = torch.mean(neg_z_scores, dim=-1).unsqueeze(-1).unsqueeze(-1)
#                     neg_global_selected = neg_global_selected + self._variant_distrs[:, concept_idx:concept_idx+1, var_idx:var_idx+1] * torch.sigmoid(
#                         self.box_temparature * (neg_z_scores - self.thresholds[concept_idx, 1])
#                     )
#                 if concept_idx == 0:
#                     if self.print_counter % 50 == 0:
#                         print(f"\tTotal positive global embeddings selected for {np.mean((pos_global_selected > 0.5).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(pos_global_selected).detach().cpu().numpy(), "and a min value of", torch.min(pos_global_selected).detach().cpu().numpy(), "when min pos z-scores is =", torch.min(pos_z_scores).detach().cpu().numpy(), "and max pos z-scores is =", torch.max(pos_z_scores).detach().cpu().numpy())
#                         print(f"\tTotal negative global embeddings selected for {np.mean((neg_global_selected > 0.5).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(neg_global_selected).detach().cpu().numpy(), "and a min value of", torch.min(neg_global_selected).detach().cpu().numpy(), "when min neg z-scores is =", torch.min(neg_z_scores).detach().cpu().numpy(), "and max neg z-scores is =", torch.max(neg_z_scores).detach().cpu().numpy())
#                     self.print_counter += 1

#                 if not training:
#                     pos_global_selected = torch.where(
#                         pos_global_selected >= 0.5,
#                         torch.ones_like(pos_global_selected),
#                         torch.zeros_like(pos_global_selected),
#                     )
#                     neg_global_selected = torch.where(
#                         neg_global_selected >= 0.5,
#                         torch.ones_like(neg_global_selected),
#                         torch.zeros_like(neg_global_selected),
#                     )

#                 else:
#                     pos_forced_on = self.ood_dropout(torch.ones_like(pos_global_selected))
#                     pos_global_selected = torch.ones_like(pos_global_selected) * pos_forced_on + (1 - pos_forced_on) * pos_global_selected
#                     neg_forced_on = self.ood_dropout(torch.ones_like(neg_global_selected))
#                     neg_global_selected = torch.ones_like(neg_global_selected) * neg_forced_on + (1 - neg_forced_on) * neg_global_selected

#             else:
#                 raise ValueError(
#                     f'Unsupported selection mode "{self.selection_mode}"'
#                 )

#             if self.pooling_mode == 'concat':
#                 pos_embeddings.append(
#                     pos_global_selected * global_contexts[:, concept_idx:concept_idx+1, :self.emb_size] +
#                     (1 - pos_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size]
#                 )
#                 neg_embeddings.append(
#                     neg_global_selected * global_contexts[:, concept_idx:concept_idx+1, self.emb_size:] +
#                     (1 - neg_global_selected) * dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:]
#                 )
#             elif self.pooling_mode == 'individual_scores':
#                 pos_embeddings.append(
#                     torch.concat(
#                         [
#                             global_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
#                             dynamic_contexts[:, concept_idx:concept_idx+1, :self.emb_size],
#                         ],
#                         dim=-1,
#                     )
#                 )
#                 neg_embeddings.append(
#                     torch.concat(
#                         [
#                             global_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
#                             dynamic_contexts[:, concept_idx:concept_idx+1, self.emb_size:],
#                         ],
#                         dim=-1,
#                     )
#                 )
#             else:
#                 raise ValueError(
#                     f'Unsupported pooling mode "{self.pooling_mode}"'
#                 )

#         pos_embeddings = torch.concat(pos_embeddings, dim=1)
#         neg_embeddings = torch.concat(neg_embeddings, dim=1)

#         return c_sem, pos_embeddings, neg_embeddings, extra_outputs



#     def _extra_losses(
#         self,
#         x,
#         y,
#         c,
#         y_pred,
#         c_sem,
#         c_pred,
#         competencies=None,
#         prev_interventions=None,
#     ):
#         loss = 0.0

#         if self.kl_loss_weight and (self._dynamic_contexts is not None):
#             for concept_idx in range(self.n_concepts):
#                 pos_pred_mean = torch.mean(self._dynamic_contexts[:, concept_idx, :self.emb_size], dim=0)
#                 pos_pred_std = torch.std(self._dynamic_contexts[:, concept_idx, :self.emb_size], dim=0)
#                 pos_pred_distr = torch.distributions.normal.Normal(
#                     pos_pred_mean,
#                     pos_pred_std,
#                 )
#                 for variant_idx in range(self.n_concept_variants):
#                     variant_probs = self._variant_distrs[:, concept_idx, variant_idx]
#                     if self.attention_fn == 'sigmoid':
#                         selected = variant_probs >= 0.5
#                     else:
#                         selected = variant_probs >= 1/self.n_concept_variants

#                     target_distr = torch.distributions.normal.Normal(
#                         self.concept_embeddings[concept_idx, variant_idx, 0, :],
#                         torch.exp(self.concept_log_devs[concept_idx, variant_idx, 0, :]),
#                     )
#                     loss += self.kl_loss_weight * (selected * torch.distributions.kl.kl_divergence(target_distr, pos_pred_distr).sum()).mean()/self.n_concepts
#             self._dynamic_contexts = None
#             self._variant_distrs = None

#         return loss










################################################################################
## Same as a above but the distributions are learnt from the dynammic context
################################################################################

class ProbConceptEmbeddingModel(IntAwareConceptEmbeddingModel):
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
        initial_log_variances=None,
        fixed_variances=False,
        attention_fn='softmax',
        ood_dropout_prob=0,
        pooling_mode='concat',
        selection_mode='z_score',
        kl_loss_weight=0,
        box_temparature=1,
        threshold=None,
        learnable_concept_embs=True,
        shared_attn_module=False,
        global_above_thresh=True,
    ):
        self._construct_c2y_model = False
        super(ProbConceptEmbeddingModel, self).__init__(
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

        # Set up any attention model needed if we have multiple variants per
        # concept
        self.shared_attn_module = shared_attn_module
        if self.n_concept_variants > 1:
            out_feats = list(
                self.pre_concept_model.modules()
            )[-1].out_features
            if self.shared_attn_module:
                self.attn_model = torch.nn.Sequential(
                    torch.nn.Linear(
                        out_feats + self.n_concepts,
                        # (2*self.emb_size) + self.n_concepts,
                        # 64,
                        self.n_concept_variants,
                    ),
                    # torch.nn.LeakyReLU(),
                    # torch.nn.Linear(
                    #     64,
                    #     32,
                    # ),
                    # torch.nn.Linear(
                    #     32,
                    #     self.n_concept_variants,
                    # ),
                )
            else:
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
            if self.shared_attn_module:
                self.attn_model = lambda x: torch.ones(x.shape[0], self.n_concept_variants).to(x.device)
            else:
                self.attn_model = [
                    lambda x: torch.ones(x.shape[0], self.n_concept_variants).to(x.device)
                    for _ in range(self.n_concepts)
                ]

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

            self.c2y_model = self._individual_score_mix
            self.global_c2y_model = lambda x: self._individual_score_mix(x, only_global=True)
        else:
            raise ValueError(
                f'Unsupported pooling mode "{self.pooling_mode}"'
            )

        self.kl_loss_weight = kl_loss_weight


        self.box_temparature = box_temparature
        if threshold is None:
            self.thresholds = torch.nn.Parameter(
                torch.normal(
                    torch.zeros(self.n_concepts, self.n_concept_variants),
                    torch.ones(self.n_concepts, self.n_concept_variants),
                ),
                requires_grad=True,
            )
        else:
            self.thresholds = threshold * torch.ones(self.n_concepts, self.n_concept_variants)


        self._dynamic_contexts = None
        self._variant_distrs = None
        self.learnable_concept_embs = learnable_concept_embs
        self.fallback_concept_embs = torch.nn.Parameter(
            torch.normal(
                torch.zeros(self.n_concepts, 2, self.emb_size),
                torch.ones(self.n_concepts, 2, self.emb_size),
            ),
            requires_grad=self.learnable_concept_embs,
        )

        self.concept_context_log_devs = torch.nn.Parameter(
            torch.normal(
                torch.zeros(self.n_concepts, self.n_concept_variants, 2*emb_size),
                torch.ones(self.n_concepts, self.n_concept_variants, 2*emb_size),
            ),
            requires_grad=True,
        )

        self.concept_context_means = torch.nn.Parameter(
            torch.normal(
                torch.zeros(self.n_concepts, self.n_concept_variants, 2*emb_size),
                torch.ones(self.n_concepts, self.n_concept_variants, 2*emb_size),
            ),
            requires_grad=True,
        )
        self.global_above_thresh = global_above_thresh

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
            dynamic_contexts = []
            global_contexts = []
            variant_distrs = []

            # First predict all the concept probabilities
            for concept_idx in range(self.n_concepts):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[concept_idx]
                dynamic_context = self._generate_dynamic_concept(pre_c, concept_idx=concept_idx)
                dynamic_contexts.append(torch.unsqueeze(dynamic_context, dim=1))

                # distr has shape (B, n_concept_variants)
                if self.shared_attn_module:
                    # distr_logits = self.attn_model(dynamic_context)
                    distr_logits = self.attn_model(
                        torch.concat(
                            [pre_c, torch.nn.functional.one_hot(torch.tensor([concept_idx for _ in range(dynamic_context.shape[0])]).to(dynamic_context.device), num_classes=self.n_concepts)],
                            dim=-1,
                    ))
                else:
                    distr_logits = self.attn_model[concept_idx](pre_c)
                if self.attention_fn == 'sigmoid':
                    distr = torch.sigmoid(
                        self.temperature * distr_logits
                    )

                elif self.attention_fn == 'softmax':
                    distr = torch.softmax(
                        self.temperature * distr_logits,
                        dim=-1,
                    )

                elif self.attention_fn == 'soft_gumbel':
                    distr = torch.nn.functional.gumbel_softmax(
                        self.temperature * distr_logits,
                        dim=-1,
                        hard=False,
                    )

                elif self.attention_fn == 'hard_gumbel':

                    distr = torch.nn.functional.gumbel_softmax(
                        self.temperature * distr_logits,
                        dim=-1,
                        hard=True,
                    )
                else:
                    raise ValueError(
                        f'Unrecognized attention_fn "{self.attention_fn}".'
                    )
                if training and (not self.learnable_concept_embs):
                    self._update_concept_vector(
                        distr=distr,
                        concept_idx=concept_idx,
                    )
                global_context_pos = self.fallback_concept_embs[concept_idx:concept_idx+1, 0, :].expand(pre_c.shape[0], -1)
                global_context_neg = self.fallback_concept_embs[concept_idx:concept_idx+1, 1, :].expand(pre_c.shape[0], -1)
                global_context = torch.concat(
                    [global_context_pos, global_context_neg],
                    dim=-1,
                )
                global_contexts.append(torch.unsqueeze(global_context, dim=1))
                variant_distrs.append(distr.unsqueeze(1))
            dynamic_contexts = torch.cat(dynamic_contexts, axis=1)
            global_contexts = torch.cat(global_contexts, axis=1)
            latent = dynamic_contexts, global_contexts, variant_distrs
        else:
            dynamic_contexts, global_contexts, variant_distrs = latent

        self._variant_distrs = torch.concat(variant_distrs, dim=1)

        # Now we can compute all the probabilites!
        c_sem = []
        c_logits = []
        self._dynamic_contexts = dynamic_contexts

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
        for concept_idx in range(self.n_concepts):
            if self.selection_mode == 'z_score':
                embs = dynamic_contexts[:, concept_idx, :]
                global_selected = 0
                cumm_zcores = 0
                for var_idx in range(self.n_concept_variants):
                    mean = self.concept_context_means[concept_idx:concept_idx+1, var_idx, :].expand(dynamic_contexts.shape[0], -1)
                    std = torch.exp(
                        self.concept_context_log_devs[concept_idx:concept_idx+1, var_idx, :].expand(dynamic_contexts.shape[0], -1)
                    )
                    z_scores = torch.abs(embs - mean) / std
                    z_scores = torch.mean(z_scores, dim=-1).unsqueeze(-1).unsqueeze(-1)
                    if self.global_above_thresh:
                        global_selected = global_selected + self._variant_distrs[:, concept_idx:concept_idx+1, var_idx:var_idx+1] * torch.sigmoid(
                            self.box_temparature * (z_scores - self.thresholds[concept_idx, var_idx])
                        )

                    else:
                        global_selected = global_selected + self._variant_distrs[:, concept_idx:concept_idx+1, var_idx:var_idx+1] * torch.sigmoid(
                            self.box_temparature * (self.thresholds[concept_idx, var_idx] - z_scores)
                        )
                    cumm_zcores = cumm_zcores + self._variant_distrs[:, concept_idx:concept_idx+1, var_idx:var_idx+1] * z_scores

                if concept_idx in [0, 10, 20] and (self.print_counter % 25 == 0):
                    print(f"\t(concept {concept_idx}) Total global embeddings selected for {np.mean((global_selected > 0.5).detach().cpu().numpy())*100:.2f}% samples with a max value of ", torch.max(global_selected).detach().cpu().numpy(), "and a min value of", torch.min(global_selected).detach().cpu().numpy(), "when min z-scores is =", torch.min(cumm_zcores).detach().cpu().numpy(), "and max z-scores is =", torch.max(cumm_zcores).detach().cpu().numpy())
                    print(f"\t\t(concept {concept_idx}) Variants selected {np.mean(self._variant_distrs[:, concept_idx, :].detach().cpu().numpy(), 0)}")

                if not training:
                    global_selected = torch.where(
                        global_selected >= 0.5,
                        torch.ones_like(global_selected),
                        torch.zeros_like(global_selected),
                    )
                    pass

                else:
                    forced_on = self.ood_dropout(torch.ones_like(global_selected))
                    global_selected = torch.ones_like(global_selected) * forced_on + (1 - forced_on) * global_selected

                pos_global_selected = global_selected
                neg_global_selected = global_selected
            else:
                raise ValueError(
                    f'Unsupported selection mode "{self.selection_mode}"'
                )

            if self.pooling_mode == 'concat':
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

        self.print_counter += 1
        pos_embeddings = torch.concat(pos_embeddings, dim=1)
        neg_embeddings = torch.concat(neg_embeddings, dim=1)

        return c_sem, pos_embeddings, neg_embeddings, extra_outputs


    def _update_concept_vector(self, distr, concept_idx, decay=0.9):
        for variant_idx in range(self.n_concept_variants):
            mixed_values = (
                distr[:, variant_idx:variant_idx+1] * self.concept_context_means[concept_idx:concept_idx+1, variant_idx, :].expand(distr.shape[0], -1)
            ).mean(0).detach()

            new_pos_value = mixed_values[:self.emb_size]
            new_neg_value = mixed_values[self.emb_size:]
            self.fallback_concept_embs[concept_idx, 0, :] = (
                decay * self.fallback_concept_embs[concept_idx, 0, :] +
                (1 - decay) * new_pos_value
            )

            self.fallback_concept_embs[concept_idx, 1, :] = (
                decay * self.fallback_concept_embs[concept_idx, 1, :] +
                (1 - decay) * new_neg_value
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

        if self.kl_loss_weight and (self._dynamic_contexts is not None):
            for concept_idx in range(self.n_concepts):
                pred_mean = torch.mean(self._dynamic_contexts[:, concept_idx, :], dim=0)
                pred_std = torch.std(self._dynamic_contexts[:, concept_idx, :], dim=0)
                pred_distr = torch.distributions.normal.Normal(
                    pred_mean,
                    pred_std,
                )
                for variant_idx in range(self.n_concept_variants):
                    variant_probs = self._variant_distrs[:, concept_idx, variant_idx]
                    # if self.attention_fn == 'sigmoid':
                    #     selected = variant_probs >= 0.5
                    # else:
                    #     selected = variant_probs >= 1/self.n_concept_variants
                    selected = variant_probs

                    target_distr = torch.distributions.normal.Normal(
                        self.concept_context_means[concept_idx, variant_idx, :],
                        torch.exp(self.concept_context_log_devs[concept_idx, variant_idx, :]),
                    )
                    loss += self.kl_loss_weight * (selected * torch.distributions.kl.kl_divergence(target_distr, pred_distr).sum()).mean()/self.n_concepts
            self._dynamic_contexts = None
            self._variant_distrs = None

        return loss



