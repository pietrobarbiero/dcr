import numpy as np
import pytorch_lightning as pl
import torch

from torchvision.models import resnet50

from cem.models.cbm import ConceptBottleneckModel
import cem.train.utils as utils
from cem.metrics.accs import compute_accuracy



################################################################################
## OUR MODEL
################################################################################


class AdversarialConceptBottleneckModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=0.01,
        task_loss_weight=1,

        extra_dims=0,
        bool=False,
        sigmoidal_prob=True,
        sigmoidal_extra_capacity=True,
        bottleneck_nonlinear=None,
        output_latent=False,

        x2c_model=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_layers=None,

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
        use_concept_groups=False,

        top_k_accuracy=None,

        # New:
        discriminator_layers=[],
        discriminator_loss_weight=1,
        interleave_steps=5,
    ):
        """
        Constructs a Concept Embedding Model (CEM) as defined by
        Espinosa Zarlenga et al. 2022.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CEM.
        :param int emb_size: The size of each concept embedding. Defaults to 16.
        :param float training_intervention_prob: RandInt probability. Defaults
            to 0.25.
        :param str embedding_activation: A valid nonlinearity name to use for the
            generated embeddings. It must be one of [None, "sigmoid", "relu",
            "leakyrelu"] and defaults to "leakyrelu".
        :param Bool shared_prob_gen: Whether or not weights are shared across
            all probability generators. Defaults to True.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the CEM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: A generator function
            for the latent code generator model that takes as an input the size
            of the latent code before the concept embedding generators act (
            using an argument called `output_dim`) and returns a valid Pytorch
            Module that maps this CEM's inputs to the latent space of the
            requested size.

        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float learning_rate:  Learning rate used for optimization.
            Default is 0.01.
        :param float weight_decay: The weight decay factor used during
            optimization. Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with
            n_tasks elements indicating the weights assigned to each output
            class during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.

        :param List[float] active_intervention_values: A list of n_concepts
            values to use when positively intervening in a given concept (i.e.,
            setting concept c_i to 1 would imply setting its corresponding
            predicted concept to active_intervention_values[i]). If not given,
            then we will assume that we use `1` for all concepts. This
            parameter is important when intervening in CEMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts
            values to use when negatively intervening in a given concept (i.e.,
            setting concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts.
        :param Callable[(np.ndarray, np.ndarray, np.ndarray), np.ndarray] intervention_policy:
            An optional intervention policy to be used when intervening on a
            test batch sample x (first argument), with corresponding true
            concepts c (second argument), and true labels y (third argument).
            The policy must produce as an output a list of concept indices to
            intervene (in batch form) or a batch of binary masks indicating
            which concepts we will intervene on.

        :param List[int] top_k_accuracy: List of top k values to report accuracy
            for during training/testing when the number of tasks is high.
        """
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        self.output_interventions = output_interventions
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        if x2c_model is not None:
            # Then this is assumed to be a module already provided as
            # the input to concepts method
            self.x2c_model = x2c_model
        else:
            self.x2c_model = c_extractor_arch(
                output_dim=(n_concepts + extra_dims)
            )

        # Now construct the label prediction model
        if c2y_model is not None:
            # Then this method has been provided to us already
            self.c2y_model = c2y_model
        else:
            # Else we construct it here directly
            units = [n_concepts + extra_dims] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)

        # Intervention-specific fields/handlers:
        if active_intervention_values is not None:
            self.active_intervention_values = torch.FloatTensor(
                active_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.active_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]
            ) * (
                5.0 if not sigmoidal_prob else 1.0
            )
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.FloatTensor(
                inactive_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.inactive_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]
            ) * (
                -5.0 if not sigmoidal_prob else 0.0
            )

        # For legacy purposes, we wrap the model around a torch.nn.Sequential
        # module
        self.sig = torch.nn.Sigmoid()
        if sigmoidal_extra_capacity:
            # Keeping this for backwards compatability
            bottleneck_nonlinear = "sigmoid"
        if bottleneck_nonlinear == "sigmoid":
            self.bottleneck_nonlin = torch.nn.Sigmoid()
        elif bottleneck_nonlinear == "leakyrelu":
            self.bottleneck_nonlin = torch.nn.LeakyReLU()
        elif bottleneck_nonlinear == "relu":
            self.bottleneck_nonlin = torch.nn.ReLU()
        elif (bottleneck_nonlinear is None) or (
            bottleneck_nonlinear == "identity"
        ):
            self.bottleneck_nonlin = lambda x: x
        else:
            raise ValueError(
                f"Unsupported nonlinearity '{bottleneck_nonlinear}'"
            )

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                pos_weight=task_class_weights
            )
        )
        self.bool = bool
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.extra_dims = extra_dims
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        self.sigmoidal_prob = sigmoidal_prob
        self.sigmoidal_extra_capacity = sigmoidal_extra_capacity
        self.use_concept_groups = use_concept_groups


        # NEW STUFF
        self.mode = 'cbm' # ['cbm', 'discriminator_warmup', 'joint']
        self._cbm_turn = True
        self.current_steps = torch.nn.Parameter(
            torch.IntTensor([0]),
            requires_grad=False,
        )
        self.discriminator_loss_weight = discriminator_loss_weight
        self.interleave_steps = interleave_steps
        units = [extra_dims] + (discriminator_layers or []) + [n_concepts]
        layers = []
        for i in range(1, len(units)):
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())
        layers.append(torch.nn.Sigmoid())
        self.discriminator = torch.nn.Sequential(*layers)

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
        )
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]

        # Then we fix block the gradient from propagating into the CBM's stuff

        if self.mode == 'cbm':
            cbm_turn = True
        elif self.mode == 'discriminator_warmup':
            cbm_turn = False
        elif self.mode == 'joint':
            if train:
                step = int(self.current_steps.detach())
                if step % self.interleave_steps == 0:
                    self._cbm_turn = not self._cbm_turn
            self.current_steps += 1
            if (not train) or self._cbm_turn:
                # Then time to train just the CBM
                # notice that during inference, we simply always use the CBM
                # as that what we care about at the end
                cbm_turn = True
                # Freeze discriminator's weights
                for param in self.discriminator.parameters():
                    param.requires_grad = False
                # Unfreeze generator's weights
                for param in self.x2c_model.parameters():
                    param.requires_grad = True
            else:
                cbm_turn = False
                # Unfreeze generator's weights
                for param in self.discriminator.parameters():
                    param.requires_grad = True
                # Freeze generator's weights
                for param in self.x2c_model.parameters():
                    param.requires_grad = False

        if cbm_turn:
            if self.task_loss_weight != 0:
                task_loss = self.loss_task(
                    y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                    y,
                )
                task_loss_scalar = task_loss.detach()
            else:
                task_loss = 0
                task_loss_scalar = 0
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
                "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
            }
            if self.mode == 'joint':
                # Then time to add the discriminator loss here!
                residual = c_logits[:, self.n_concepts:]
                c_discriminator_preds = self.discriminator(residual)
                discriminator_loss = -self.loss_concept(c_discriminator_preds, c)
                loss += self.discriminator_loss_weight * discriminator_loss
                (discr_c_accuracy, discr_c_auc, discr_c_f1), _ = compute_accuracy(
                    c_pred=c_discriminator_preds,
                    y_pred=None,
                    c_true=c,
                    y_true=y,
                )
                result['discr_c_accuracy'] = discr_c_accuracy
                result['discr_c_auc'] = discr_c_auc
                result['discr_loss'] = discriminator_loss.detach()
                result['current_steps'] = self.current_steps.detach()
            result.update({
                "loss": loss.detach(),
            })
        else:
            # Then we fix block the gradient from propagating into the CBM's stuff
            residual = c_logits[:, self.n_concepts:].detach()
            c_discriminator_preds = self.discriminator(residual)
            loss = self.loss_concept(c_discriminator_preds, c)
            (c_accuracy, c_auc, c_f1), (_, _, _) = compute_accuracy(
                c_pred=c_discriminator_preds,
                y_pred=None,
                c_true=c,
                y_true=y,
            )
            result = {
                "c_accuracy": c_accuracy,
                "c_auc": c_auc,
                "c_f1": c_f1,
                "loss": loss.detach(),
            }
        if self.mode == 'joint':
            result['mode'] = 2 + int(not cbm_turn)
        else:
            result['mode'] = {'cbm': 0, 'discriminator_warmup':1, 'joint': 2}[self.mode]
        return loss, result
