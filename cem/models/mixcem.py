import numpy as np
import pytorch_lightning as pl
import scipy
import sklearn.metrics
import torch
import math

from torchvision.models import resnet50

from cem.metrics.accs import compute_accuracy
from cem.models.cem import ConceptEmbeddingModel
import cem.train.utils as utils

def log(x):
    return torch.log(x + 1e-6)

def Gaussian_CDF(x): #change from erf function
    return 0.5 * (1. + torch.erf(x / np.sqrt(2.)))


def _binary_entropy(probs):
    return -probs * torch.log2(probs) - (1 - probs) * torch.log2(1 - probs)







class TransposeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.transpose(input, -2, -1)

class VectorLinear(torch.nn.Module):

    def __init__(
        self,
        in_embs: int,
        emb_size: int,
        dim,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_embs = in_embs
        self.dim = dim
        self.emb_size = emb_size
        self.weight = torch.nn.Parameter(
            torch.empty((in_embs, 1), **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((emb_size,), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.emb_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.dim in [1, -2]:
            result = input[:, 0, :] * self.weight[0, 0]
            for i in range(1, self.in_embs):
                result += input[:, i, :] * self.weight[i, 0]
        else:
            result = input[:, :, 0] * self.weight[0, 0]
            for i in range(1, self.in_embs):
                result += input[:, :, i] * self.weight[i, 0]
        if self.bias is not None:
            result += self.bias
        return result

    def extra_repr(self) -> str:
        return f"in_embs={self.in_embs}, emb_size={self.emb_size}, bias={self.bias is not None}"




################################################################################
## Final Version
################################################################################

class MixingConceptEmbeddingModel(ConceptEmbeddingModel):
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

        # Residual stuff
        residual_scale=1,
        conditional_residual=False,
        residual_layers=None,
        per_concept_residual=False,
        sigmoidal_residual=False,
        shared_per_concept_residual=False,
        residual_deviation=0,

        # Mixing
        additive_mixing=False,

        mix_ground_truth_embs=True,
        normalize_embs=False,
        cov_mat=None,

        fixed_embeddings=False,
        initial_concept_embeddings=None,
        use_cosine_similarity=False,
        use_linear_emb_layer=False,
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
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        if c_extractor_arch == "identity":
            self.pre_concept_model = lambda x: x
        else:
            self.pre_concept_model = c_extractor_arch(output_dim=None)
        if self.pre_concept_model == "identity":
            c_extractor_arch = "identity"
            self.pre_concept_model = lambda x: x

        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        self.mix_ground_truth_embs = mix_ground_truth_embs
        self.normalize_embs = normalize_embs
        self.intervention_task_discount = intervention_task_discount
        self.intermediate_task_concept_loss = intermediate_task_concept_loss
        if cov_mat is None:
            cov_mat = np.eye(n_concepts, dtype=np.float32)
        self.cov_mat = cov_mat
        self.L = torch.tensor(
            scipy.linalg.cholesky(self.cov_mat, lower=True).astype(
                np.float32
            )
        )

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
        self.contrastive_loss_fn = torch.nn.CosineEmbeddingLoss(
            margin=0.0,
            size_average=None,
            reduce=None,
            reduction='mean',
        )
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
        self.shared_emb_generator = True
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
            if self.shared_emb_generator:
                if len(self.concept_emb_generators) == 0:
                    self.concept_emb_generators.append(
                        torch.nn.Sequential(*[
                            torch.nn.Linear(
                                list(
                                    self.pre_concept_model.modules()
                                )[-1].out_features,
                                emb_size,
                            ),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(
                                emb_size,
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
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            emb_size,
                            emb_size,
                        ),
                        emb_act,
                    ])
                )

        self.use_linear_emb_layer = use_linear_emb_layer
        self.additive_mixing = additive_mixing
        if c2y_model is None:
            # Else we construct it here directly
            if self.use_linear_emb_layer:
                units = [self.emb_size] + (c2y_layers or []) + [n_tasks]
                layers = [
                    torch.nn.Unflatten(-1, (self.n_concepts, self.emb_size)),
                    VectorLinear(in_embs=self.n_concepts, emb_size=self.emb_size, dim=1),
                    torch.nn.LeakyReLU(),
                ]
                for i in range(1, len(units)):
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                    if i != len(units) - 1:
                        layers.append(torch.nn.LeakyReLU())
                self.c2y_model = torch.nn.Sequential(*layers)
            else:
                units = [
                    self.n_concepts * self.emb_size
                    if not additive_mixing else self.emb_size
                ] + (c2y_layers or []) + [n_tasks]
                layers = []
                for i in range(1, len(units)):
                    layers.append(torch.nn.LeakyReLU())
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model


        self.sig = torch.nn.Sigmoid()

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights, reduction='none')
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights,
                reduction='none',
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
        self._current_pred_concepts = None

        self.use_cosine_similarity = use_cosine_similarity
        if self.use_cosine_similarity:
            self._cos_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        else:
            self._cos_similarity = None

        # Residual handling
        self.per_concept_residual = per_concept_residual
        self.residual_scale = residual_scale
        self.conditional_residual = conditional_residual
        self.sigmoidal_residual = sigmoidal_residual
        self.shared_per_concept_residual = shared_per_concept_residual
        self.residual_deviation = residual_deviation
        if self.per_concept_residual and self.shared_per_concept_residual:
            units = [
                emb_size * 2

            ] + (residual_layers or []) + [
                emb_size if per_concept_residual else n_tasks
            ]
        else:
            units = [
                (emb_size * (n_concepts + 1) if conditional_residual else emb_size)
                if not additive_mixing
                else (emb_size * 2 if conditional_residual else emb_size)

            ] + (residual_layers or []) + [
                n_concepts * emb_size if per_concept_residual
                else n_tasks
            ]
        layers = []
        for i in range(1, len(units)):
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Linear(units[i-1], units[i]))

        if sigmoidal_residual:
            layers.append(torch.nn.Sigmoid())
        self.residual_model = torch.nn.Sequential(*layers)

        # Black-box label predictor
        self.warmup_mode = False
        self.bypass_label_predictor = torch.nn.Linear(
            emb_size,
            n_tasks,
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
        return 0

    def _relaxed_multi_bernoulli_sample(self, probs, temperature=1, idx=None):
        # Sample from a standard Gaussian first to perform the
        # reparameterization trick
        shape = (probs.shape[0],)
        epsilon = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)).to(
            probs.device
        )
        u = Gaussian_CDF(epsilon)
        return torch.sigmoid(
            1.0/temperature * (
                log(probs) - log(1. - probs) + log(u) - log(1. - u)
            )
        )

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
        out_embeddings=None,
        pred_concepts=None,
        pre_c=None,
        **kwargs,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            if self.additive_mixing:
                bottleneck = torch.zeros(
                    (pred_concepts.shape[0], self.emb_size)
                ).to(pred_concepts.device)
                for i in range(self.n_concepts):
                    bottleneck += pred_concepts[:, i, :]
            else:
                bottleneck = torch.flatten(
                    pred_concepts,
                    start_dim=1,
                )
            return prob, intervention_idxs, bottleneck
        if intervention_idxs is None:
            intervention_idxs = torch.zeros((c_true.shape[0], self.n_concepts))

        # First mixed trained concepts
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        output = prob * (1 - intervention_idxs) + intervention_idxs * c_true

        # [Shape: (1, n_concepts, emb_size)]
        pos_anchors = self.concept_embeddings[:, 0, :].unsqueeze(0)
        if self.normalize_embs:
            pos_anchors = torch.nn.functional.normalize(
                pos_anchors,
                dim=1,
            )
        # [Shape: (1, n_concepts, emb_size)]
        neg_anchors = self.concept_embeddings[:, 1, :].unsqueeze(0)
        if self.normalize_embs:
            neg_anchors = torch.nn.functional.normalize(
                neg_anchors,
                dim=1,
            )
        # [Shape: (B, n_concepts, 1)]
        extended_intervention_idxs = torch.unsqueeze(
            intervention_idxs,
            dim=-1,
        )
        # [Shape: (B, n_concepts, 1)]
        extended_c_true = c_true.unsqueeze(-1)
        # [Shape: (B, n_concepts, emb_size)]
        ground_truth_anchors = (
            pos_anchors * extended_c_true +
            (1 - extended_c_true) * neg_anchors
        )
        # [Shape: (B, n_concepts, emb_size)]
        pred_concepts = (
            (1 - extended_intervention_idxs) * pred_concepts +
            extended_intervention_idxs * ground_truth_anchors
        )
        if train:
            self._current_pred_concepts = pred_concepts


        # Then time to mix!
        if self.additive_mixing:
            bottleneck = torch.zeros(
                (pred_concepts.shape[0], self.emb_size)
            ).to(pred_concepts.device)
            for i in range(self.n_concepts):
                bottleneck += pred_concepts[:, i, :]
        else:
            bottleneck = torch.flatten(
                pred_concepts,
                start_dim=1,
            )
        return output, intervention_idxs, bottleneck

    def _distance_metric(self, neg_anchor, pos_anchor, latent):
        if self.use_cosine_similarity:
            neg_sim = self._cos_similarity(neg_anchor, latent)
            pos_sim = self._cos_similarity(pos_anchor, latent)
            return pos_sim - neg_sim

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
            c_sem = []
            pred_concepts = []

            # First predict all the concept probabilities
            for i, concept_emb_generator in enumerate(
                self.concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                projected_space = concept_emb_generator(pre_c)
                if self.normalize_embs:
                    projected_space = torch.nn.functional.normalize(
                        projected_space,
                        dim=-1,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_pos_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 0, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_pos_emb = torch.nn.functional.normalize(
                        anchor_concept_pos_emb,
                        dim=-1,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_neg_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 1, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_neg_emb = torch.nn.functional.normalize(
                        anchor_concept_neg_emb,
                        dim=-1,
                    )
                # [Shape: (B)]
                prob = self.sig(
                    self.contrastive_scale[i] * self._distance_metric(
                        neg_anchor=anchor_concept_neg_emb,
                        pos_anchor=anchor_concept_pos_emb,
                        latent=projected_space,
                    )
                )
                # [Shape: (B, 1)]
                prob = torch.unsqueeze(prob, dim=-1)
                if self.mix_ground_truth_embs:
                    mixed_embs = (
                        prob * anchor_concept_pos_emb +
                        (1 - prob) * anchor_concept_neg_emb
                    )
                else:
                    mixed_embs = projected_space

                c_sem.append(prob)
                pred_concepts.append(
                    torch.unsqueeze(mixed_embs, dim=1)
                )
            c_sem = torch.cat(c_sem, dim=-1)
            pred_concepts = torch.cat(pred_concepts, dim=1)
            latent = c_sem, pred_concepts, projected_space
        else:
            c_sem, pred_concepts, projected_space = latent
        if training:
            self._current_pred_concepts = pred_concepts
        return c_sem, None, None, {
            "pred_concepts": pred_concepts,
            "pre_c": pre_c,
            "projected_space": projected_space,
        }

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
        probs, intervention_idxs, bottleneck = self._after_interventions(
            c_sem,
            pos_embeddings=pos_embs,
            neg_embeddings=neg_embs,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
            **out_kwargs
        )
        if len(bottleneck.shape) > 2:
            bottleneck = bottleneck.view((bottleneck.shape[0], -1))
        projected_space = out_kwargs['projected_space']
        if self.residual_deviation:
            # Then add some noise!
            projected_space = projected_space + torch.normal(
                0,
                torch.ones_like(projected_space) * self.residual_deviation,
            )
        if self.conditional_residual and not (
            self.per_concept_residual and self.shared_per_concept_residual
        ):
            projected_space = torch.concat(
                [bottleneck, projected_space],
                dim=-1
            )

        if self.warmup_mode:
            y_pred = self.bypass_label_predictor(projected_space)
        elif self.per_concept_residual and self.shared_per_concept_residual:
            self._pre_residual_acts = bottleneck
            bottleneck = bottleneck.view(
                bottleneck.shape[0],
                self.n_concepts,
                self.emb_size,
            )

            for i in range(self.n_concepts):
                updated_input = torch.concat(
                    [projected_space, bottleneck[:, i, :]],
                    dim=-1,
                )
                res = self.residual_scale * self.residual_model(updated_input)
                bottleneck[:, i, :] += res
            bottleneck = bottleneck.view((bottleneck.shape[0], -1))
            y_pred = self.c2y_model(bottleneck)
        else:
            self._pre_residual_acts = bottleneck
            residual = self.residual_scale * self.residual_model(
                projected_space
            )
            if self.per_concept_residual:
                y_pred = self.c2y_model(bottleneck + residual)
            else:
                y_pred = self.c2y_model(bottleneck) + residual
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
                latent = (latent or tuple([])) + out_kwargs['latent']
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
        for param in self.residual_model.parameters():
            param.requires_grad = False

    def unfreeze_residual(self):
        for param in self.residual_model.parameters():
            param.requires_grad = True

    def freeze_label_predictor(self):
        for param in self.c2y_model.parameters():
            param.requires_grad = False

    def unfreeze_label_predictor(self):
        for param in self.c2y_model.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.pre_concept_model.parameters():
            param.requires_grad = False
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

        int_probs = []
        og_int_probs = self.training_intervention_prob
        if isinstance(self.training_intervention_prob, list) and train:
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
            current_task_loss = self.task_loss_weight * self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
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
        if (self.intermediate_task_concept_loss != 0) and (
            self._pre_residual_acts is not None
        ):
            # Then we will also include a task loss term for when the residual
            # is not used at all
            pre_residual_y_logits = self.c2y_model(self._pre_residual_acts)
            task_loss += self.intermediate_task_concept_loss * self.loss_task(
                (
                    pre_residual_y_logits if pre_residual_y_logits.shape[-1] > 1
                    else pre_residual_y_logits.reshape(-1)
                ),
                y,
            )
            self._pre_residual_acts = None

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



