"""
Simple API for evaluating a suite of metrics of interest given a model or
its predictions.

Taken from: Espinosa Zarlenga et al., "Efficient Bias Mitigation Without Privileged Information." ECCV (2024).
"""

import numpy as np
import scipy.special

import cem.metrics.task_metrics as task_metrics

def normalize(y_pred):
    """ Normalizes predictions to represent probabilities if such predictions
    are provided as logits.

    Args:
        y_pred (np.ndarray): Predictions with shape [batch, 1] or [batch], for
            binary outputs, or with shape [batch, n_classes] for categorical
            outputs.

    Returns:
        np.ndarray: an array with the same shape as y_pred whose entrie
            represent probabilities for each output class.
    """
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1):
        # This is the binary case! Let's check if it is in [0, 1]
        if (np.min(y_pred) < 0) or (np.max(y_pred) > 1):
            y_pred = scipy.special.expit(y_pred)
        return y_pred
    # Else the multi-label case
    if (np.min(y_pred) < 0) or (np.max(y_pred) > 1):
        y_pred = scipy.special.softmax(y_pred, axis=-1)
    return y_pred


def test_metrics(
    y_true,
    y_pred,
    metrics,
    n_labels,
    attributes=None,
    **kwargs,
):
    """Computes a set of metrics defined in the list `metrics` over the given
    set of predictions `y_pred` and corresponding targets `y_true`.

    Args:
        y_true (np.ndarray): Target classes for each of the samples. We expect
            this to be an np.ndarray with shape [B], where B is the number of
            samples in the batch, such that each element is an integer in the
            set {0, 1, ..., n_classes - 1}.
        y_pred (np.ndarray): Predicted probabilities for each of the samples. If
            the task is binary, then we expect a numpy array containing numbers
            in [0, 1] with shape [B, 1] or [B], where B is the number of samples
            in the batch. If the task is multi-class/categorical, then we
            expect y_pred to have shape [B, n_classes].
        metrics (list[str]): A list of strings representing valid supported
            metrics such as ["accuracy", "auc", "f1", "precision", "recall"].
        n_labels (int): Number of class labels one can possibly find in the
            y_true.
        attributes (np.ndarray, optional): Binary tensor indicating the
            attributes that are present in each of the samples. We expect
            this to be an np.ndarray with shape [B, n_attrs], where B is the
            number of samples in the batch and n_attrs is the number of known
            attributes in the dataset. If not given, then we will assume that
            we do not have any known attributes in this dataset.

    Raises:
        ValueError: If an unsupported metric is provided in the list of metrics.

    Returns:
        list[float]: List of metric scores corresponding to the metrics
            requested by the list `metrics` in the same order.
    """
    # First normalize the prediction so that they can be interpreted as
    # probabilities
    y_pred = normalize(y_pred)

    # Next iterate over all metrics, matching the name with the corresponding
    # call and adding the computed value to the list.
    results = []
    for metric in metrics:
        used_metric_name = metric.lower().strip()
        if used_metric_name in ["accuracy", "acc"]:
            results.append(task_metrics.accuracy(
                y_true,
                y_pred,
                threshold=kwargs.get('accuracy_threshold', 0.5),
            ))
        elif used_metric_name in ["multilabel_accuracy", "multilabel_acc"]:
            results.append(task_metrics.multilabel_accuracy(
                y_true,
                y_pred,
                threshold=kwargs.get('accuracy_threshold', 0.5),
            ))
        elif used_metric_name in [
            "train_class_weighted_accuracy",
            "train_class_weighted_acc",
        ]:
            class_distributions = kwargs['train_class_distributions']
            sample_weights = np.zeros_like(y_true)
            for label in np.unique(y_true):
                sample_weights[y_true == label] = class_distributions[label]
            results.append(task_metrics.train_weighted_accuracy(
                y_true,
                y_pred,
                sample_weights=sample_weights,
                threshold=kwargs.get('accuracy_threshold', 0.5),
            ))
        elif used_metric_name in [
            "train_group_weighted_accuracy",
            "train_group_weighted_acc",
        ]:
            assert attributes is not None, (
                "Cannot compute group weighted accuracy if we are not "
                "provided with group attributes"
            )
            classes = np.unique(y_true)
            sample_weights = np.zeros_like(y_true)
            if kwargs.get(
                'train_group_weighted_acc_check_subgroups',
                False,
            ):
                class_distributions = kwargs['train_group_class_distributions']
                # all_vals = np.array(list(class_distributions.values()))
                # scaling_factor = np.min(all_vals[all_vals != 0])
                for label in classes:
                    for group_idx in range(attributes.shape[-1]):
                        for pos_val in [0, 1]:
                            selected = np.logical_and(
                                y_true == label,
                                attributes[:, group_idx] == pos_val,
                            )
                            sample_weights[selected] = class_distributions[
                                (label, group_idx, pos_val)
                            ] #scaling_factor
            else:
                group_distributions = kwargs['train_group_distributions']
                for group_idx in range(attributes.shape[-1]):
                    for pos_val in [0, 1]:
                        selected = attributes[:, group_idx] == pos_val
                        sample_weights[selected] = group_distributions[
                            (group_idx, pos_val)
                        ]
            results.append(task_metrics.train_weighted_accuracy(
                y_true,
                y_pred,
                sample_weights=sample_weights,
                threshold=kwargs.get('accuracy_threshold', 0.5),
            ))
        elif used_metric_name in [
            "mean_multilabel_accuracy",
            "mean_multilabel_acc",
        ]:
            results.append(task_metrics.mean_multilabel_accuracy(
                y_true,
                y_pred,
                threshold=kwargs.get('accuracy_threshold', 0.5),
            ))
        elif used_metric_name == "auc":
            results.append(task_metrics.auc(
                y_true,
                y_pred,
                multi_class=kwargs.get('auc_multi_class', 'ovo'),
            ))
        elif used_metric_name == "f1":
            results.append(task_metrics.f1(
                y_true,
                y_pred,
                average=kwargs.get('f1_average', 'micro'),
            ))
        elif used_metric_name == "precision":
            results.append(task_metrics.precision(
                y_true,
                y_pred,
                average=kwargs.get('precision_average', 'micro'),
            ))
        elif used_metric_name == "recall":
            results.append(task_metrics.recall(
                y_true,
                y_pred,
                average=kwargs.get('recall_average', 'micro'),
            ))
        else:
            raise ValueError(
                f'Unsupported metric {metric}!'
            )

    return results

