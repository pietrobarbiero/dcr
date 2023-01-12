import torch
import numpy as np
from torch.nn.functional import one_hot


def load_data(dataset, fold, train_epochs):
    if dataset in ['cub', 'celeba']:
        c = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_Adaptive_NoProbConcat_resnet34_fold_{fold}/test_embedding_semantics_on_epoch_{train_epochs}.npy')
        c_emb = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_Adaptive_NoProbConcat_resnet34_fold_{fold}/test_embedding_vectors_on_epoch_{train_epochs}.npy')
        y = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/y_val.npy')
    else:
        c = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_semantics_on_epoch_{train_epochs}.npy')
        c_emb = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_vectors_on_epoch_{train_epochs}.npy')
        y = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/y_val.npy')

    c = torch.FloatTensor(c>0.5)
    c_emb = torch.FloatTensor(c_emb)
    c_emb = c_emb.reshape(c_emb.shape[0], c.shape[1], -1)
    y = torch.LongTensor(y)
    y1h = one_hot(y).float()

    # generate random train and test masks
    np.random.seed(42)
    train_mask = set(np.random.choice(np.arange(c.shape[0]), int(c.shape[0] * 0.8), replace=False))
    test_mask = set(np.arange(c.shape[0])) - train_mask
    train_mask = torch.LongTensor(list(train_mask))
    test_mask = torch.LongTensor(list(test_mask))

    # define train set
    c_emb_train = c_emb[train_mask]
    c_scores_train = c[train_mask]
    y_train = y1h[train_mask]

    # define test set
    c_emb_test = c_emb_test_all = c_emb[test_mask]
    c_scores_test = c_scores_test_all = c[test_mask]
    y_test = y1h[test_mask]

    if dataset == 'celeba':
        # in celeba we first re-define the task to simulate an OOD setting
        y_train = one_hot(torch.sum(c_scores_train[:, :2], dim=1).long()).float()
        y_test = one_hot(torch.sum(c_scores_test[:, :2], dim=1).long()).float()

        # and then we simulate a real-world scenario where the first two concepts are missing
        c_emb_test = c_emb_test_all[:, 1:]
        c_scores_test = c_scores_test_all[:, 1:]

    return c_emb_train, c_scores_train, y_train, c_emb_test, c_scores_test, y_test
