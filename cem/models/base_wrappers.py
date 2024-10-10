"""
A base wrapper is a wrapper around a Pytorch module that handles all the
evaluation of the model (as provided by some metrics of interest) as well as
it allows one to easily train the same model using different losses and
optimizer configurations.

The goal of these objects is to abstract the loss compputation and gradient
update from the state of the Pytorch object so that these two things
can be kept separately.

Taken from: Espinosa Zarlenga et al., "Efficient Bias Mitigation Without Privileged Information." ECCV (2024).

"""
import numpy as np
import pytorch_lightning as pl
import torch

import cem.metrics.test as evaluation


def _id(x):
    # For pickeable purposes
    return x

################################################################################
## CONFIG LOADERS
################################################################################

def loss_from_config(
    loss_config,
    task_class_weights=None,
    functional=False,
):
    """Generates a torch loss function from the given config dictionary.

    Args:
        loss_config (dict): A valid dictionary containing the configuration of
             the desired loss. This dictionary must have a value for its
             `name` key, representing the name of the loss function to use, and
             may contain optional keys such as `from_logits` and
             `class_weighted`.
        task_class_weights (np.ndarray, optional): Optional set of class weights
            to be used in the loss if we wish to weight samples differently
            across classes. Defaults to None.
        functional (bool, False): If set, then we will output Pytorch's
            functional form of the loss rather than the Module-based output.
            By default this method will always output a Pytorch loss Module
            rather than a function.

    Raises:
        ValueError: if provided with an invalid or unsupported loss name.

    Returns:
        torch.nn.Loss: A valid Torch loss function corresponding to the provided
            config file.
    """
    weight_loss = None
    if "name" not in loss_config:
        raise ValueError(
            "We expect each loss config to at least have a valid 'name' "
            "value in it."
        )
    if loss_config.get("class_weighted", False):
        weight_loss = torch.FloatTensor(task_class_weights)
    if loss_config["name"].lower() in ["cross_entropy", "ce"]:
        if functional:
            return lambda input, target, weight: \
                torch.nn.functional.cross_entropy(
                    input=input,
                    target=target,
                    weight=weight,
                    reduction=loss_config.get('reduction', 'mean'),
                )
        return torch.nn.CrossEntropyLoss(
            weight=weight_loss,
            reduction=loss_config.get('reduction', 'mean'),
        )
    elif loss_config["name"].lower() in [
        "binary_cross_entropy",
        "binary_ce",
        "bce",
    ]:
        if loss_config.get('from_logits', False):
            if functional:
                return lambda input, target, weight: \
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        input=torch.flatten(input),
                        target=torch.flatten(target),
                        weight=weight,
                        reduction=loss_config.get('reduction', 'mean'),
                    )
            return torch.nn.BCEWithLogitsLoss(
                weight=weight_loss,
                reduction=loss_config.get('reduction', 'mean'),
            )

        if functional:
            return lambda input, target, weight: \
                torch.nn.functional.binary_cross_entropy(
                    input=torch.flatten(input),
                    target=torch.flatten(target),
                    weight=weight,
                    reduction=loss_config.get('reduction', 'mean'),
                )
        return torch.nn.BCELoss(
            weight=weight_loss,
            reduction=loss_config.get('reduction', 'mean'),
        )

    # Else complain!
    raise ValueError(
        f'We do not support loss {loss_config["name"]} yet!'
    )

class EvalOnlyWrapperModule(pl.LightningModule):
    """Wrapper around a Pytorch Module that enables us to evaluate, and ONLY
    evaluate, this model using Pytorch Lightning's API.
    """
    def __init__(
        self,
        model,
        n_labels,
        output_activation=None,
        metrics=["accuracy"],
        metrics_kwargs=None,
        logits_name="logits",
    ):
        """Evaluation only wrapper for a Pytorch Module handling metric
        computation and evaluation.

        Args:
            model (torch.Module): Pytorch model/module which we wish to
                evaluate.
            n_labels (int): Number of output labels of the provided model.
            output_activation (str, optional): If not None, then it must
                be a string representing an activation function which we wish
                to apply to the output of `model` before computing
                any evaluation metrics. If None, then this is an identity
                activation. It must be one of ["sigmoid", "softmax"]. Defaults
                to None.
            metrics (list[str], optional): If provided this is a list of metric
                names which we wish to evaluate for the provided model. For a
                list of allowed metric names, please refer to the docs for
                bias_transfer.evaluation.test.  Defaults to ["accuracy"].
            metrics_kwargs (dict, optional): Optional dictionary of keyword
                arguments to be provided to the evaluation metrics in the
                `metrics` list. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        self.metrics = metrics
        self.metrics_kwargs = metrics_kwargs or {}
        self.wrapped_model = model
        self.n_labels = n_labels
        self.logits_name = logits_name

        # Set up any output activations which we want to add on top of the given
        # model
        if output_activation == "sigmoid":
            self.output_activation = torch.nn.Sigmoid()
        elif output_activation == "softmax":
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif output_activation is None:
            # Then we assume the model already outputs a sigmoidal vector
            self.output_activation = _id
        else:
            raise ValueError(
                f'Unsupported output activation {output_activation}'
            )

    def activation_for_probs(self, x):
        if len(x.shape) > 1 and (x.shape[-1] > 1):
            return torch.softmax(x, dim=-1)
        return torch.sigmoid(x.reshape(-1))

    def forward(self, x):
        outputs = self._access_model()(x)
        if isinstance(outputs, (tuple, list)):
            logits = self.output_activation(
                outputs[0]
            )
            return [logits] + outputs[1:]
        elif isinstance(outputs, dict):
            outputs = self.output_activation(
                outputs[self.logits_name]
            )
        return self.output_activation(outputs)

    def _access_model(self):
        return self.wrapped_model

    def _unpack_batch(self, batch, batch_idx):
        """Helper function to unpack a given batch of samples. Assumes the
        first two elements are always the sample `x` and the label `y`.
        If there is a third element in the batch, then it assumes that we have
        been provided with binary group annotations in the form of a binary
        vector.

        This method always outputs a tuple (x, y, attributes). If no attributes
        have been provided by the dataset/task, then they are returned as None.
        """
        if not isinstance(batch, (list, tuple)):
            # Then no labels have been provided and this is the pure inference
            # mode!
            return x, None, None
        if len(batch) == 1:
            x = batch[0]
            y = None
            attributes = None
        elif len(batch) == 2:
            x, y = batch
            attributes = None
        elif len(batch) >= 3:
            x, y, attributes = batch[:3]
        else:
            raise ValueError(
                f'Unsupported batch with {len(batch)} elements in it!'
            )
        if isinstance(y, (list, tuple)):
            # Then we will assume the first element is always the label
            # of interest
            if len(y) > 1:
                y, attributes = y[:2]
            else:
                y = y[0]
        return x, y, attributes

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, _ = self._unpack_batch(batch, batch_idx)
        return self(x)

    def _run_step(self, batch, batch_idx, train=False):
        x, y, attributes = self._unpack_batch(batch, batch_idx)
        y_logits = self(x)
        if isinstance(y_logits, (list, tuple)):
            # Then we always assume it is the first element
            y_logits = y_logits[0]
        elif isinstance(y_logits, dict):
            # Then we assume there is an output named logits!
            y_logits = y_logits[self.logits_name]
        # compute all metrics of interest
        if y is not None:
            eval_result = evaluation.test_metrics(
                y_true=y.cpu().detach().numpy(),
                y_pred=self.activation_for_probs(y_logits).cpu().detach().numpy(),
                attributes=(
                    attributes.cpu().detach().numpy()
                    if not (attributes is None) else None
                ),
                metrics=self.metrics,
                n_labels=self.n_labels,
                model=self.wrapped_model,
                **self.metrics_kwargs,
            )
            step_metrics = {
                key: val for (key, val) in zip(self.metrics, eval_result)
            }
        else:
            step_metrics = {}
        return y_logits, step_metrics

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            if (
                (isinstance(val, (int, float))) or
                (len(val.shape) == 0)
            ):
                self.log("val_" + name, val, prog_bar=("accuracy" in name), sync_dist=True)
        return {
            "val_" + key: val
            for key, val in result.items()
        }

    def test_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True, sync_dist=True)
        return result

class WrapperModule(EvalOnlyWrapperModule):
    """Wrapper around a Pytorch Module that enables the configuration of
    an arbitrary optimizer and loss function to train this model using Pytorch
    Lightning's API.
    """
    def __init__(
        self,
        model,
        optimizer_config,
        loss_config,
        n_labels,
        output_activation=None,
        metrics=["accuracy"],
        task_class_weights=None,
        metrics_kwargs=None,
        functional_loss=False,
        prog_bar_vars=['acc'],
        logits_name='logits',
    ):
        """Wrapper for easily training a Pytorch Module by handling metric
        computation, evaluation, optimizer setup, and loss computation.

        Args:
            model (torch.Module): Pytorch model/module which we wish to
                train/evaludate.
            optimizer_config (dict): a valid optimizer config containing the
                key "optimizer" which specifies the kind of optimizer to
                use (e.g., "adam" or "sgd"). The rest of keys can specify
                a learning rate schedule (via the "lr_scheduler" key) as well
                as the hyperparameters of the optimizer itself such as the
                learning rate ("learning_rate"), momentum ("momentum"),
                weight decay ("weight_decay"), etc.
            loss_config (dict): A dictionary containing the parameters required
                to configure the training loss for the provided model. This
                config should contain keys such as "name", describing name of
                the loss to be used (e.g., "cross_entropy",
                "binary_cross_entropy", etc).
            n_labels (int): Number of output labels of the provided model.
            output_activation (str, optional): If not None, then it must
                be a string representing an activation function which we wish
                to apply to the output of `model` before computing
                any evaluation metrics. If None, then this is an identity
                activation. It must be one of ["sigmoid", "softmax"]. Defaults
                to None.
            metrics (list[str], optional): If provided this is a list of metric
                names which we wish to evaluate for the provided model. For a
                list of allowed metric names, please refer to the docs for
                bias_transfer.evaluation.test.  Defaults to ["accuracy"].
            task_class_weights (_type_, optional): _description_. Defaults to None.
            metrics_kwargs (dict, optional): Optional dictionary of keyword
                arguments to be provided to the evaluation metrics in the
                `metrics` list. Defaults to None.
            functional_loss (bool, optional): whether or not to use Pytorch's
                function version of the loss when we instantiate the loss or
                using the corresponding Pytorch loss module. Defaults to False.
            prog_bar_vars (list[str], optional): List of metric names in the
                list `metrics` which we wish to include in the progress
                bar shown whilst training the wrapped model. If provided, then
                all metrics containing as a substring one of the strings in this
                list will be printed in the progress bar during training.
                Defaults to ['acc'].
        """
        super().__init__(
            model=model,
            output_activation=output_activation,
            metrics=metrics,
            n_labels=n_labels,
            metrics_kwargs=metrics_kwargs,
            logits_name=logits_name,
        )
        self.loss_task = loss_from_config(
            loss_config,
            task_class_weights=task_class_weights,
            functional=functional_loss,
        )
        self.optimizer_config = optimizer_config
        self.prog_bar_vars = prog_bar_vars

    def _run_step(self, batch, batch_idx, train=False, output_logits=False):
        _, y, _ = self._unpack_batch(batch, batch_idx)
        y_logits, step_metrics = super()._run_step(
            batch,
            batch_idx,
            train=train,
        )
        loss = self.loss_task(
            y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
            y,
        )
        step_metrics["loss"] = loss.detach()
        if output_logits:
            return loss, y_logits, step_metrics
        return loss, step_metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, _ = self._unpack_batch(batch, batch_idx)
        outputs = self._access_model()(x)
        other_outputs = {}
        if isinstance(outputs, (tuple, list)):
            y_logits = self.output_activation(
                outputs[0]
            )
        elif isinstance(outputs, dict):
            y_logits = self.output_activation(
                outputs[self.logits_name]
            )
            other_outputs = {
                key: val for key, val in outputs.items() if key != self.logits_name
            }
        else:
            y_logits = outputs
        y_logits = self.output_activation(y_logits)

        if y is not None:
            # Then we can also predict a loss value here!
            loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            return dict(
                logits=y_logits,
                loss=loss,
                **other_outputs,
            )

        # Otherwise we have only the logits to produce
        return dict(
            logits=y_logits,
            **other_outputs,
        )

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            if (
                (isinstance(val, (int, float))) or
                (len(val.shape) == 0)
            ):
                self.log(
                    name,
                    val,
                    prog_bar=(np.any(list(map(
                        lambda x: name != "loss" and (x in name),
                        self.prog_bar_vars,
                    )))),
                    sync_dist=True,
                )
        result.pop("loss", None)
        return {
            "loss": loss,
            "log": {
                key: val for (key, val) in result.items()
            },
        }

    def test_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            if (
                (isinstance(val, (int, float))) or
                (len(val.shape) == 0)
            ):
                self.log("test_" + name, val, prog_bar=True, sync_dist=True)
        return result['loss']

    def configure_optimizers(self):
        if "optimizer" not in self.optimizer_config:
            raise ValueError(
                f"Expected key `optimizer` to be in the provided optimizer "
                f"config."
            )
        if self.optimizer_config["optimizer"].lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_config.get("learning_rate", 1e-3),
                weight_decay=self.optimizer_config.get("weight_decay", 0),
            )
        elif self.optimizer_config["optimizer"].lower() == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.optimizer_config.get("learning_rate", 1e-3),
                momentum=self.optimizer_config.get("momentum", 0),
                weight_decay=self.optimizer_config.get("weight_decay", 0),
            )
        else:
            raise ValueError(
                f'Unsupported optimizer {self.optimizer_config["optimizer"]}'
            )
        if "lr_scheduler" in self.optimizer_config:
            lr_scheduler_config = self.optimizer_config["lr_scheduler"]
            if lr_scheduler_config["type"] == "reduce_on_plateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer
                )
            else:
                raise ValueError(
                    f"Unsupported learning rate "
                    f"scheduler {lr_scheduler_config['type']}"
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "loss",
            }

        else:
            return {
                "optimizer": optimizer,
                "monitor": "loss",
            }
