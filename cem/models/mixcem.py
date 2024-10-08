import numpy as np
import pytorch_lightning as pl
import scipy
import sklearn.metrics
import torch
import math

from torchvision.models import resnet50

from cem.metrics.accs import compute_accuracy
from cem.models.cem import ConceptEmbeddingModel
from cem.models.intcbm import IntAwareConceptEmbeddingModel
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

def dot(x, y):
    return (x * y).sum(dim=-1).unsqueeze(-1)

def _orthogonal_projection(latent_space, embs):
    # Latent space has shape (B, m)
    # embs has shape (k, m)
    # Output should have shape (B, k, m)
    result = []
    used_pred_concepts = embs.expand(latent_space.shape[0], -1, -1)
    for i in range(used_pred_concepts.shape[1]):
        y_y = dot(used_pred_concepts[:, i, :], used_pred_concepts[:, i, :])
        x_y = dot(latent_space, used_pred_concepts[:, i, :])
        orth_proj = latent_space - (x_y * used_pred_concepts[:, i, :]) / (y_y + 1e-8)
        orth_proj = torch.nn.functional.normalize(orth_proj, dim=-1)
        result.append(orth_proj.unsqueeze(1))
    return torch.concat(result, dim=1)

class MixingConceptEmbeddingModel(IntAwareConceptEmbeddingModel):
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
        use_residual=True,
        residual_scale=1,
        learnable_residual_scale=False,
        conditional_residual=False,
        residual_layers=None,
        per_concept_residual=False,
        sigmoidal_residual=False,
        shared_per_concept_residual=False,
        learn_residual_embeddings=False,
        noise_residual_embedings=False,
        residual_deviation=0,
        residual_norm_loss=0,
        residual_norm_metric=1,
        residual_scale_norm_metric=1,
        residual_scale_reg=0,
        residual_model_weight_l2_reg=0,
        dynamic_residual=False,
        sigmoidal_residual_scale=False,
        warmup_mode=False,
        include_bypass_model=False,

        # Mixing
        bottleneck_pooling='concat',
        learnable_distance_metric=False,
        learnable_prob_model=False,
        use_latent_space=False,

        mix_ground_truth_embs=True,
        normalize_embs=False,

        fixed_embeddings=False,
        initial_concept_embeddings=None,
        use_cosine_similarity=False,
        use_linear_emb_layer=False,
        fixed_scale=None,

        # Extra capacity
        extra_capacity=0,
        extra_capacity_dropout_prob=0,
        orthogonal_extra_capacity=False,

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
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.bottleneck_pooling = bottleneck_pooling
        self.orthogonal_extra_capacity = orthogonal_extra_capacity
        if orthogonal_extra_capacity:
            extra_capacity = n_concepts * emb_size
        self.extra_capacity = extra_capacity
        if extra_capacity:
            assert bottleneck_pooling == 'concat', 'Currently only support extra capcity when using concat pooling'
            if orthogonal_extra_capacity:
                self.extra_capacity_residual = lambda latent, probs: (
                    probs.unsqueeze(-1) * _orthogonal_projection(latent, self.concept_embeddings[:, 0, :]) +
                    (1 - probs.unsqueeze(-1)) * _orthogonal_projection(latent, self.concept_embeddings[:, 1, :])
                ).view(latent.shape[0], -1)
            else:
                self._extra_capacity_residual =  torch.nn.Sequential(*[
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, self.extra_capacity),
                    torch.nn.Dropout(p=extra_capacity_dropout_prob),
                    torch.nn.Sigmoid(),
                ])
                self.extra_capacity_residual = lambda x, *args: self._extra_capacity_residual(x)
        else:
            self.extra_capacity_residual = None
        assert self.bottleneck_pooling in ['concat', 'additive', 'mean', 'weighted_mean', 'per_class_mixing', 'per_class_mixing_shared'], (
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
            self.bottleneck_size = emb_size * n_concepts + self.extra_capacity
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
        self.residual_model_weight_l2_reg = residual_model_weight_l2_reg

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
        if c2y_model is None:
            # Else we construct it here directly
            if 'per_class_mixing' in self.bottleneck_pooling:
                self.downstream_label_weights = torch.nn.Parameter(
                    torch.rand((n_tasks, self.n_concepts)),
                    requires_grad=True,
                )
                if self.bottleneck_pooling == 'per_class_mixing_shared':
                    self.class_embeddings = torch.nn.Parameter(
                        torch.rand((n_tasks, self.emb_size)),
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
                    self.c2y_model = torch.nn.ModuleList()
                    for i in range(n_tasks):
                        units = [self.emb_size] + (c2y_layers or []) + [1]
                        layers = []
                        for i in range(1, len(units)):
                            layers.append(torch.nn.LeakyReLU())
                            layers.append(torch.nn.Linear(units[i-1], units[i]))
                        self.c2y_model.append(torch.nn.Sequential(*layers))
            elif self.use_linear_emb_layer:
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
                units = [self.bottleneck_size] + (c2y_layers or []) + [n_tasks]
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
            torch.nn.CrossEntropyLoss(weight=task_class_weights) #, reduction='none')
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights,
                # reduction='none',
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

        self.use_cosine_similarity = use_cosine_similarity
        if self.use_cosine_similarity:
            self._cos_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        else:
            self._cos_similarity = None


        self.learnable_distance_metric = learnable_distance_metric
        if self.learnable_distance_metric:
            units = [
                2 * emb_size
            ] + [128, 64, 1]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != (len(units) - 1):
                    layers.append(torch.nn.LeakyReLU())
            self.distance_model = torch.nn.Sequential(*layers)

        self.use_latent_space = use_latent_space
        if learnable_prob_model:
            units = [
                emb_size if use_latent_space else 2 * emb_size
            ] + [1]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != (len(units) - 1):
                    layers.append(torch.nn.LeakyReLU())
            self.learnable_prob_model = torch.nn.Sequential(*layers)
        else:
            self.learnable_prob_model = None


        # Black-box label predictor
        self.warmup_mode = warmup_mode
        if self.warmup_mode or include_bypass_model:
            self.bypass_label_predictor = torch.nn.Linear(
                emb_size,
                n_tasks,
            )

        # Residual handling
        if not use_residual:
            # Then that's it for setup!
            self.residual_model = None

        self.per_concept_residual = per_concept_residual
        self.learnable_residual_scale = learnable_residual_scale
        self.conditional_residual = conditional_residual
        self.sigmoidal_residual = sigmoidal_residual
        self.shared_per_concept_residual = shared_per_concept_residual
        self.residual_deviation = residual_deviation
        self.residual_norm_loss = residual_norm_loss
        self.residual_norm_metric = residual_norm_metric
        self.residual_scale_norm_metric = residual_scale_norm_metric
        self.residual_scale_reg = residual_scale_reg
        self.sigmoidal_residual_scale = sigmoidal_residual_scale
        self._training_residual = True
        self._current_residuals = []

        if self.learnable_residual_scale:
            if self.per_concept_residual:
                if residual_scale is not None:
                    self.residual_scale = torch.nn.Parameter(
                        residual_scale * torch.ones((self.n_concepts,)),
                        requires_grad=True,
                    )
                else:
                    self.residual_scale = torch.nn.Parameter(
                        torch.rand((self.n_concepts,)),
                        requires_grad=True,
                    )
            else:
                if residual_scale is not None:
                    self.residual_scale = torch.nn.Parameter(
                        residual_scale * torch.ones((self.n_tasks,)),
                        requires_grad=True,
                    )
                else:
                    self.residual_scale = torch.nn.Parameter(
                        torch.rand((self.n_tasks,)),
                        requires_grad=True,
                    )
        else:
            self.residual_scale = (
                residual_scale if residual_scale is not None else 1
            )

        self.learn_residual_embeddings = learn_residual_embeddings
        self.noise_residual_embedings = noise_residual_embedings
        self.dynamic_residual = dynamic_residual
        if self.learn_residual_embeddings and (not self.dynamic_residual):
            self.residual_embeddings = torch.nn.Parameter(
                torch.rand((self.n_concepts, 2, self.emb_size)),
                requires_grad=True,
            )
        if self.dynamic_residual:
            units = [
                emb_size
            ] + (residual_layers or []) + [emb_size * 2]
            self.residual_model = torch.nn.ModuleList()

            for _ in range(self.n_concepts):
                layers = []
                for i in range(1, len(units)):
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                    if i != (len(units) - 1):
                        layers.append(torch.nn.LeakyReLU())

                layers.append(torch.nn.Unflatten(-1, (2, emb_size)))
                self.residual_model.append(torch.nn.Sequential(*layers))
        elif self.per_concept_residual:
            if self.shared_per_concept_residual:
                assert conditional_residual, (
                    'conditional_residual needs to be True if '
                    'shared_per_concept_residual is True'
                )
                if self.learn_residual_embeddings:
                    units = [
                            (emb_size * 2)
                            if conditional_residual else emb_size

                        ] + (residual_layers or []) + [1]
                    layers = []
                    for i in range(1, len(units)):
                        layers.append(torch.nn.Linear(units[i-1], units[i]))
                        if i != (len(units) - 1):
                            layers.append(torch.nn.LeakyReLU())
                    self.residual_model = torch.nn.Sequential(*layers)

                else:
                    units = [
                        emb_size * 2

                    ] + (residual_layers or []) + [
                        emb_size if per_concept_residual else n_tasks
                    ]
                    layers = []
                    for i in range(1, len(units)):
                        layers.append(torch.nn.LeakyReLU())
                        layers.append(torch.nn.Linear(units[i-1], units[i]))

                    if sigmoidal_residual:
                        layers.append(torch.nn.Sigmoid())

                    self.residual_model = torch.nn.Sequential(*layers)
            elif self.learn_residual_embeddings:
                units = [
                        (emb_size * 2)
                        if conditional_residual else emb_size

                    ] + (residual_layers or []) + [1]
                self.residual_model = torch.nn.ModuleList()

                for _ in range(self.n_concepts):
                    layers = []
                    for i in range(1, len(units)):
                        layers.append(torch.nn.Linear(units[i-1], units[i]))
                        if i != (len(units) - 1):
                            layers.append(torch.nn.LeakyReLU())

                    if sigmoidal_residual:
                        layers.append(torch.nn.Sigmoid())
                    self.residual_model.append(torch.nn.Sequential(*layers))
            else:
                units = [
                    (2 * emb_size) if conditional_residual else emb_size

                ] + (residual_layers or []) + [emb_size]
                self.residual_model = torch.nn.ModuleList()
                for _ in range(self.n_concepts):
                    layers = []
                    for i in range(1, len(units)):
                        layers.append(torch.nn.LeakyReLU())
                        layers.append(torch.nn.Linear(units[i-1], units[i]))

                    if sigmoidal_residual:
                        layers.append(torch.nn.Sigmoid())
                    self.residual_model.append(
                        torch.nn.Sequential(*layers)
                    )

        elif self.learn_residual_embeddings:
            units = [
                    (emb_size + self.bottleneck_size)
                    if conditional_residual else emb_size

                ] + (residual_layers or []) + [n_concepts]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != (len(units) - 1):
                    layers.append(torch.nn.LeakyReLU())
            if sigmoidal_residual:
                layers.append(torch.nn.Sigmoid())
            self.residual_model = torch.nn.Sequential(*layers)

        else:
            units = [
                    (emb_size + self.bottleneck_size)
                    if conditional_residual else emb_size

                ] + (residual_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.LeakyReLU())
                layers.append(torch.nn.Linear(units[i-1], units[i]))

            if sigmoidal_residual:
                layers.append(torch.nn.Sigmoid())
            self.residual_model = torch.nn.Sequential(*layers)




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
        if self.residual_norm_loss and self._current_residuals and (
            self.task_loss_weight > 0
        ):
            for res_norm in self._current_residuals:
                loss += self.residual_norm_loss * torch.mean(res_norm) / len(self._current_residuals)
            self._current_residuals = []
        if self.learnable_residual_scale and self.per_concept_residual and (
            self.residual_scale_reg
        ):
            if self.sigmoidal_residual_scale:
                loss += torch.norm(self.sig(self.residual_scale), p=self.residual_scale_norm_metric)
            else:
                loss += torch.norm(self.residual_scale, p=self.residual_scale_norm_metric)

        if self.residual_model_weight_l2_reg and (self.residual_model is not None):
            total_sum = 0.0
            total_elems = 0
            for param in self.residual_model.parameters():
                total_sum += torch.pow(param, 2).sum()
                total_elems += np.prod(param.shape)
            total_sum = self.residual_model_weight_l2_reg * total_sum
            loss += total_sum/(total_elems if total_elems else 1)
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

    def _construct_c2y_input(
        self,
        pos_embeddings,
        neg_embeddings,
        probs,
        **task_loss_kwargs,
    ):
        pred_concepts = (
            pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                neg_embeddings * (
                    1 - torch.unsqueeze(probs, dim=-1)
                )
            )
        )
        return self._make_bottleneck(
            pred_concepts=pred_concepts,
            projected_space=task_loss_kwargs['projected_space'],
            probs=probs,
            train=True,
        )

    def _make_bottleneck(
        self,
        pred_concepts,
        projected_space,
        probs,
        train=False,
    ):
        self._pre_residual_acts = pred_concepts.view(pred_concepts.shape[-1], -1)
        used_pred_concepts = pred_concepts
        if self.dynamic_residual:
            if self.residual_norm_loss:
                self._current_residuals = []
            # Then we actually introduce some residuals here!
            used_pred_concepts = torch.zeros_like(pred_concepts)
            for i in range(self.n_concepts):
                # Size (B, 2, m)
                pos_res, neg_res, scale = self._make_anchor_residual(
                    projected_space=projected_space,
                    concept_idx=i,
                )
                res_emb = probs[:, i:i+1] * pos_res + (1 - probs[:, i:i+1]) * neg_res
                if self.residual_norm_loss:
                    self._current_residuals.append(torch.norm(res_emb, p=self.residual_norm_metric, dim=-1))
                used_global_emb = pred_concepts[:, i, :]
                used_residual = res_emb
                if self.normalize_embs:
                    # used_global_emb = torch.nn.functional.normalize(
                    #     pred_concepts[:, i, :],
                    #     dim=-1,
                    # )
                    used_residual = torch.nn.functional.normalize(res_emb, dim=-1)
                used_pred_concepts[:, i, :] = (
                    used_global_emb  + scale * used_residual
                )
        elif (self.residual_model is not None) and self.per_concept_residual:
            if self.residual_norm_loss:
                self._current_residuals = []
            if self.shared_per_concept_residual and self._training_residual:
                # Then we actually introduce some residuals here!
                used_pred_concepts = (
                    torch.zeros_like(pred_concepts) +
                    pred_concepts
                )
                if self.learn_residual_embeddings:
                    res_embs = probs.unsqueeze(-1) * self.residual_embeddings[:, 0, :].unsqueeze(0) + (1 - probs.unsqueeze(-1)) * self.residual_embeddings[:, 1, :].unsqueeze(0)
                for i in range(self.n_concepts):
                    updated_input = torch.concat(
                        [projected_space, pred_concepts[:, i, :]],
                        dim=-1,
                    )
                    if self.learnable_residual_scale:
                        scale = self.residual_scale[i]
                    else:
                        scale = self.residual_scale
                    if self.sigmoidal_residual_scale:
                        scale = self.sig(scale)
                        if train:
                            scale = self._relaxed_multi_bernoulli_sample(scale)
                    res = scale * self.residual_model(
                        updated_input
                    )
                    if self.residual_norm_loss:
                        self._current_residuals.append(torch.norm(res, p=self.residual_norm_metric, dim=-1))
                    if self.learn_residual_embeddings:
                        res_emb = res_embs[:, i, :]
                        if self.noise_residual_embedings:
                            # Then the residual scale is treated as a log scale
                            # for the normal distribution
                            res = torch.exp(res)
                            if train:
                                res_emb = res_emb + torch.normal(
                                    0,
                                    torch.ones_like(res_emb),
                                )
                        used_pred_concepts[:, i, :] += (
                            # Shape: (B, m)
                            pred_concepts[:, i, :]  +
                            # Shape: (1, m)
                            res * res_emb
                        )
                    else:
                        used_pred_concepts[:, i, :] += res
            elif (self.residual_model is not None) and self.learn_residual_embeddings and self._training_residual:
                # Then we actually introduce some residuals here!
                used_pred_concepts = torch.zeros_like(pred_concepts)
                res_embs = probs.unsqueeze(-1) * self.residual_embeddings[:, 0, :].unsqueeze(0) + (1 - probs.unsqueeze(-1)) * self.residual_embeddings[:, 1, :].unsqueeze(0)
                for i in range(self.n_concepts):
                    if self.learnable_residual_scale:
                        scale =  self.residual_scale[i]
                    else:
                        scale = self.residual_scale

                    if self.conditional_residual:
                        updated_input = torch.concat(
                            [projected_space, pred_concepts[:, i, :]],
                            dim=-1,
                        )
                    else:
                        updated_input = projected_space
                    # Shape (B, 1)
                    res = self.residual_model[i](
                        updated_input
                    )
                    if self.sigmoidal_residual_scale:
                        scale = self.sig(scale)
                        if train:
                            scale = self._relaxed_multi_bernoulli_sample(scale)
                    res = scale * res
                    if self.residual_norm_loss:
                        self._current_residuals.append(torch.norm(res, p=self.residual_norm_metric, dim=-1))

                    res_emb = res_embs[:, i, :]
                    if self.noise_residual_embedings:
                        # Then the residual scale is treated as a log scale
                        # for the normal distribution
                        res = torch.exp(res)
                        if train:
                            res_emb = res_emb + torch.normal(
                                0,
                                torch.ones_like(res_emb),
                            )
                    used_pred_concepts[:, i, :] += (
                        # Shape: (B, m)
                        pred_concepts[:, i, :]  +
                        # Shape: (1, m)
                        res * res_emb
                    )
            elif (self.residual_model is not None) and self._training_residual:
                for i in range(self.n_concepts):
                    if self.learnable_residual_scale:
                        scale = self.residual_scale[i]
                    else:
                        scale = self.residual_scale
                    if self.sigmoidal_residual_scale:
                        scale = self.sig(scale)
                        if train:
                            scale = self._relaxed_multi_bernoulli_sample(scale)
                    if self.conditional_residual:
                        updated_input = torch.concat(
                            [projected_space, pred_concepts[:, i, :]],
                            dim=-1,
                        )
                    else:
                        updated_input = projected_space
                    res = scale * self.residual_model[i](
                        updated_input
                    )
                    if self.residual_norm_loss:
                        self._current_residuals.append(
                            torch.norm(res, p=self.residual_norm_metric, dim=-1)
                        )
                    used_pred_concepts[:, i, :] += res

        elif (self.residual_model is not None) and self.learn_residual_embeddings and self._training_residual:
            scale = self.residual_scale
            if self.sigmoidal_residual_scale:
                scale = self.sig(scale)
                if train:
                    scale = self._relaxed_multi_bernoulli_sample(scale)
            if self.conditional_residual:
                updated_input = torch.concat(
                    [projected_space, pred_concepts.view(pred_concepts.shape[0], -1)],
                    dim=-1,
                )
            else:
                updated_input = projected_space
            res =  self.residual_model(
                updated_input
            )
            # Shape: (B, k, 1)
            res = res.unsqueeze(-1)
            if self.residual_norm_loss:
                self._current_residuals.append(
                    torch.norm(res, p=self.residual_norm_metric, dim=-1)
                )
            res_embs = self.residual_embeddings[:, 0, :].unsqueeze(0) + (1 - probs) * self.residual_embeddings[:, 1, :].unsqueeze(0)
            if self.noise_residual_embedings:
                # Then the residual scale is treated as a log scale
                # for the normal distribution
                res = torch.exp(res)
                if train:
                    res_embs = res_embs + torch.normal(
                        0,
                        torch.ones_like(res_embs),
                    )
            used_pred_concepts = (
                # Shape: (B, k, m)
                pred_concepts  +
                # Shape: (B, k, m)
                res * res_embs
            )


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
            bottleneck = torch.flatten(
                used_pred_concepts,
                start_dim=1,
            )
            if self.extra_capacity_residual is not None:
                extra_capacity = self.extra_capacity_residual(projected_space, probs)
                bottleneck = torch.concat(
                    [bottleneck, extra_capacity],
                    dim=-1,
                )
        elif 'per_class_mixing' in self.bottleneck_pooling:
            bottleneck = used_pred_concepts
        else:
            raise ValueError(
                f'Unsupported bottleneck pooling "{self.bottleneck_pooling}".'
            )
        return bottleneck


    def _distance_metric(self, neg_anchor, pos_anchor, latent):

        if self.use_cosine_similarity:
            # neg_sim = self._cos_similarity(neg_anchor, latent)
            # pos_sim = self._cos_similarity(pos_anchor, latent)
            pos_projection = pos_anchor * dot(pos_anchor, latent)/(torch.norm(pos_anchor, p=2) + 1e-8)
            neg_projection = neg_anchor * dot(neg_anchor, latent)/(torch.norm(neg_anchor, p=2) + 1e-8)
            neg_dist = (neg_anchor - neg_projection).pow(2).sum(-1).sqrt()
            pos_dist = (pos_anchor - pos_projection).pow(2).sum(-1).sqrt()
            return neg_dist - pos_dist
        if self.learnable_distance_metric:
            neg_dist = (
                self.distance_model(torch.concat([neg_anchor.expand(latent.shape[0], -1), latent],  dim=-1)) +
                self.distance_model(torch.concat([latent, neg_anchor.expand(latent.shape[0], -1)],  dim=-1))
            ) / 2
            neg_dist = torch.exp(neg_dist.squeeze(-1))
            pos_dist = (
                self.distance_model(torch.concat([pos_anchor.expand(latent.shape[0], -1), latent],  dim=-1)) +
                self.distance_model(torch.concat([latent, pos_anchor.expand(latent.shape[0], -1)],  dim=-1))
            ) / 2
            pos_dist = torch.exp(pos_dist.squeeze(-1))
        else:
            neg_dist = (neg_anchor - latent).pow(2).sum(-1).sqrt()
            pos_dist = (pos_anchor - latent).pow(2).sum(-1).sqrt()
        return neg_dist - pos_dist

    def _make_anchor_residual(
        self,
        projected_space,
        concept_idx,
    ):
        if (self.residual_model is None) or (not self.learnable_prob_model):
            return 0
        res_emb = self.residual_model[concept_idx](projected_space)
        if self.learnable_residual_scale:
            scale = self.residual_scale[concept_idx]
        else:
            scale = self.residual_scale

        used_residual = res_emb
        if self.normalize_embs:
            used_residual = torch.nn.functional.normalize(res_emb, dim=-1)
        return scale * used_residual[:, 0, :], scale * used_residual[:, 1, :], scale


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
            pos_embs = []
            neg_embs = []

            # First predict all the concept probabilities
            for i, concept_emb_generator in enumerate(
                self.concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                projected_space = concept_emb_generator(pre_c)
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
                if self.learnable_prob_model is not None:
                    # [Shape: (B, 1)]
                    if self.use_latent_space:
                        prob = self.sig(
                            self.learnable_prob_model(
                                projected_space
                            )
                        )
                    else:
                        pos_res, neg_res, scale = self._make_anchor_residual(projected_space, concept_idx=i)
                        prob = self.sig(
                            self.learnable_prob_model(
                                torch.concat(
                                    [anchor_concept_pos_emb + scale * pos_res, anchor_concept_neg_emb + scale * neg_res],
                                    dim=-1
                                )
                            )
                        )
                else:
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

                anchor_concept_pos_emb = anchor_concept_pos_emb.unsqueeze(1)
                if anchor_concept_pos_emb.shape[0] == 1:
                    anchor_concept_pos_emb = anchor_concept_pos_emb.expand(
                        prob.shape[0],
                        -1,
                        -1,
                    )
                pos_embs.append(anchor_concept_pos_emb)

                anchor_concept_neg_emb = anchor_concept_neg_emb.unsqueeze(1)
                if anchor_concept_neg_emb.shape[0] == 1:
                    anchor_concept_neg_emb = anchor_concept_neg_emb.expand(
                        prob.shape[0],
                        -1,
                        -1,
                    )
                neg_embs.append(anchor_concept_neg_emb)

                c_sem.append(prob)
                pred_concepts.append(
                    torch.unsqueeze(mixed_embs, dim=1)
                )
            c_sem = torch.cat(c_sem, dim=-1)
            pos_embs = torch.cat(pos_embs, dim=1)
            neg_embs = torch.cat(neg_embs, dim=1)
            pred_concepts = torch.cat(pred_concepts, dim=1)
            latent = pre_c,  c_sem, pred_concepts, projected_space, pos_embs, neg_embs
        else:
            pre_c, c_sem, pred_concepts, projected_space, pos_embs, neg_embs = latent
        self._task_loss_kwargs.update({
            'pre_c': pre_c,
            'projected_space': projected_space,
        })
        return c_sem, pos_embs, neg_embs, {
            "pred_concepts": pred_concepts,
            "pre_c": pre_c,
            "projected_space": projected_space,
            "latent": latent,
        }


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
        projected_space = out_kwargs.pop('projected_space')
        _, intervention_idxs, bottleneck = \
            self._after_interventions(
                c_sem,
                pos_embeddings=pos_embs,
                neg_embeddings=neg_embs,
                projected_space=projected_space,
                intervention_idxs=intervention_idxs,
                c_true=c_int,
                train=train,
                competencies=competencies,
                **out_kwargs
            )

        if self.warmup_mode:
            y_pred = self.bypass_label_predictor(projected_space)
        elif 'per_class_mixing' in self.bottleneck_pooling:
            logits = []
            for task_idx in range(self.n_tasks):
                # shape (B, emb_size)
                mixed_emb = torch.sum(
                    self.downstream_label_weights.unsqueeze(0)[:, task_idx:task_idx+1, :].transpose(1, 2) * bottleneck,
                    dim=1,
                )
                if self.bottleneck_pooling == 'per_class_mixing_shared':
                    # shape (1, emb_size)
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
                    logits.append(self.c2y_model[task_idx](mixed_emb))

            if len(logits) > 1:
                y_pred = torch.concat(logits, dim=-1)
            else:
                y_pred = logits[0]

        else:
            y_pred = self.c2y_model(bottleneck)

        if (not self.warmup_mode) and (not self.per_concept_residual) and (
            not self.learn_residual_embeddings
        ) and (self.residual_model is not None):
            if self.residual_deviation:
                # Then add some noise!
                projected_space = projected_space + torch.normal(
                    0,
                    torch.ones_like(projected_space) * self.residual_deviation,
                )

            scale = self.residual_scale
            if self.sigmoidal_residual_scale:
                scale = self.sig(scale)
            if self.conditional_residual:
                projected_space = torch.concat(
                    [bottleneck.view(bottleneck.shape[0], -1), projected_space],
                    dim=-1
                )
            residual = self.residual_model(
                projected_space
            )
            if self.residual_norm_loss:
                self._current_residuals = [torch.norm(residual, p=self.residual_norm_metric, dim=-1)]
            y_pred += self.residual_scale * residual

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
        # self._training_residual = False
        if (self.residual_model is None):
            return
        for param in self.residual_model.parameters():
            param.requires_grad = False
        if self.learn_residual_embeddings:
            self.residual_embeddings.requires_grad = False
        if self.learnable_residual_scale:
            self.residual_scale.requires_grad = False

    def unfreeze_residual(self):
        # self._training_residual = True
        if (self.residual_model is None):
            return
        for param in self.residual_model.parameters():
            param.requires_grad = True
        if self.learn_residual_embeddings:
            self.residual_embeddings.requires_grad = True
        if self.learnable_residual_scale:
            self.residual_scale.requires_grad = True

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
