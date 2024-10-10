"""
Implementation of metrics for evaluating task performance in classification
tasks.

Taken from: Espinosa Zarlenga et al., "Efficient Bias Mitigation Without Privileged Information." ECCV (2024).
"""

import numpy as np
import sklearn.metrics

def make_discrete(y_pred, threshold=0.5):
    """Discretizes an array representing probabilities for each possible
    class into an array containing the labels of the prediction made for
    each sample (rather than its probability).

    Args:
        y_pred (np.ndarray): Predicted probabilities for each of the samples. If
            the task is binary, then we expect a numpy array containing numbers
            in [0, 1] with shape [B, 1] or [B], where B is the number of samples
            in the batch. If the task is multi-class/categorical, then we
            expect y_pred to have shape [B, n_classes].
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.

    Returns:
        np.ndarray: An array with shape [B] containing the class with the
            highest probability for each sample of y_pred.
    """
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1):
        # Then simply compute the actual accuracy
        y_pred = (y_pred >= threshold).astype(np.int32)
    else:
        # Else this is the multi label case
        y_pred = np.argmax(y_pred, axis=-1)
    return y_pred


def accuracy(y_true, y_pred, threshold=0.5):
    """Evaluates the mean accuracy over a batch of samples when predicting
    target labels y_true using probabilities y_pred for each of the output
    classes. y_pred may represent binary or categorical outputs.

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
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.

    Returns:
        float: mean accuracy of predicting y_true with y_pred.
    """
    return sklearn.metrics.accuracy_score(
        y_true,
        make_discrete(y_pred, threshold),
    )

def train_weighted_accuracy(y_true, y_pred, sample_weights, threshold=0.5):
    """Evaluates the weighted accuracy over a batch of samples when predicting
    target labels y_true using probabilities y_pred for each of the output
    classes. The weights are computed based on the distribution of groups or
    labels in the training set.
    y_pred may represent binary or categorical outputs.

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
        sample_weights (np.ndarry): Array of size [B] indicating the weights
            to use for each of the samples.
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.

    Returns:
        float: mean accuracy of predicting y_true with y_pred.
    """
    return sklearn.metrics.accuracy_score(
        y_true,
        make_discrete(y_pred, threshold),
        sample_weight=sample_weights,
    )

def multilabel_accuracy(y_true, y_pred, threshold=0.5):
    """Evaluates the multi-label accuracy over a batch of samples when
    predicting target multi-labels y_true using probabilities y_pred for each
    of the output labels. Each of the entries in y_pred is assumed to be a
    binary label. This method computes exact accuracy (meaning all sublabels
    must be correctly predicted for a sample to be identified as being
    correctly predicted). For a mean accuracy accross all sublabels, please
    use mean_multilabel_accuracy.

    Args:
        y_true (np.ndarray): Target multi-labels for each of the samples. We
            expect this to be a binary np.ndarray with shape [B, n_labels],
            where B is the number of samples in the batch, and n_labels is the
            number of binary labels on this task.
        y_pred (np.ndarray): Predicted probabilities for each of the samples. We
            expect a numpy array containing numbers in [0, 1] with shape
            [B, n_labels] as in y_true.
        threshold (float, optional): This is the threshold to use for binarizing
            the probabilities in y_pred. Defaults to 0.5.

    Returns:
        float: exact multi-label accuracy of predicting the labels in y_true
            with the predicted probabilities from y_pred.
    """
    return sklearn.metrics.accuracy_score(
        y_true,
        (y_pred >= threshold).astype(np.int32),
    )

def mean_multilabel_accuracy(y_true, y_pred, threshold=0.5):
    """Evaluates the mean multi-label accuracy over a batch of samples when
    predicting target multi-labels y_true using probabilities y_pred for each
    of the output labels. Each of the entries in y_pred is assumed to be a
    binary label. This method computes mean accuracy accross all sublabels for
    a given sample.

    Args:
        y_true (np.ndarray): Target multi-labels for each of the samples. We
            expect this to be a binary np.ndarray with shape [B, n_labels],
            where B is the number of samples in the batch, and n_labels is the
            number of binary labels on this task.
        y_pred (np.ndarray): Predicted probabilities for each of the samples. We
            expect a numpy array containing numbers in [0, 1] with shape
            [B, n_labels] as in y_true.
        threshold (float, optional): This is the threshold to use for binarizing
            the probabilities in y_pred. Defaults to 0.5.

    Returns:
        float: mean multi-label accuracy of predicting the labels in y_true
            with the predicted probabilities from y_pred.
    """
    acc = 0
    for task_idx in range(y_pred.shape[-1]):
        acc += sklearn.metrics.accuracy_score(
            y_true[:, task_idx],
            (y_pred[:, task_idx] >= threshold).astype(np.int32),
        )
    return acc / y_pred.shape[-1]

def balanced_accuracy(y_true, y_pred, threshold=0.5):
    """Evaluates the class-balanced mean accuracy over a batch of samples when
    predicting target labels y_true using probabilities y_pred for each of the
    output classes. y_pred may represent binary or categorical outputs.

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
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.

    Returns:
        float: mean class-balanced accuracy of predicting y_true with y_pred.
    """

    return sklearn.metrics.balanced_accuracy_score(
        y_true,
        make_discrete(y_pred, threshold),
    )

def auc(y_true, y_pred, multi_class='ovo'):
    """Evaluates the area under the receiver operating characteristic curve
    over a batch of samples when predicting target labels y_true using
    probabilities y_pred for each of the output classes. y_pred may represent
    binary or categorical outputs.

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
        multi_class (str, optional): When dealing with multi-class target labels
            this argument indicates whether we should evaluate the one-vs-one
            AUC-ROC ('ovo') or the one-vs-all AUC-ROC ('ova') to produce a
            single scalar score for the AUC.
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.

    Returns:
        float: mean area under the receiver operating characteristic curve when
            predicting y_true with y_pred.
    """
    if len(np.unique(y_true)) == 1:
        # Then we have a missing class in this batch, so we will compute
        # the accuracy instead
        return accuracy(y_true=y_true, y_pred=y_pred)
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1):
        # Then simply compute the actual accuracy
        return sklearn.metrics.roc_auc_score(
            y_true,
            y_pred,
        )

    # Else this is the multi label case
    return sklearn.metrics.roc_auc_score(
        y_true,
        y_pred,
        multi_class=multi_class,
    )

def f1(y_true, y_pred, average='micro', threshold=0.5):
    """Evaluates the f1 score over a batch of samples when predicting target
    labels y_true using probabilities y_pred for each of the output classes.
    y_pred may represent binary or categorical outputs.

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
        average (str, optional): This parameter is required for multiclass
            targets. If None, the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the data. See
            documentation for sklearn.metrics.f1_score for details.
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.

    Returns:
        float: mean F1 score when predicting y_true with y_pred.
    """
    return sklearn.metrics.f1_score(
        y_true,
        make_discrete(y_pred, threshold),
        average=average,
    )

def recall(y_true, y_pred, average='micro', threshold=0.5):
    """Evaluates the recall over a batch of samples when predicting target
    labels y_true using probabilities y_pred for each of the output classes.
    y_pred may represent binary or categorical outputs.

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
        average (str, optional): This parameter is required for multiclass
            targets. If None, the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the data. See
            documentation for sklearn.metrics.recall_score for details.
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.

    Returns:
        float: mean recall when predicting y_true with y_pred.
    """
    return sklearn.metrics.recall_score(
        y_true,
        make_discrete(y_pred, threshold),
        average=average,
    )

def precision(y_true, y_pred, average='micro', threshold=0.5):
    """Evaluates the precision over a batch of samples when predicting target
    labels y_true using probabilities y_pred for each of the output classes.
    y_pred may represent binary or categorical outputs.

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
        average (str, optional): This parameter is required for multiclass
            targets. If None, the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the data. See
            documentation for sklearn.metrics.precision_score for details.
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.

    Returns:
        float: mean precision when predicting y_true with y_pred.
    """
    return sklearn.metrics.precision_score(
        y_true,
        make_discrete(y_pred, threshold),
        average=average,
    )