# import sys
# sys.path.append('..')
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, dense_diff_pool

import numpy as np
import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything
from scipy.spatial.distance import cdist

from dcr.data.ba_shapes import load_ba_shapes, NUMBER_OF_CLASSES, load_ba_shapes_node_class

from sklearn.cluster import KMeans

import ba_shapes_model_utils
from torch_geometric.loader import DataLoader

global activation_list
activation_list = {}

def register_hooks(model):
    for name, m in model.named_modules():
            if isinstance(m, GCNConv) or isinstance(m, nn.Linear) or isinstance(m, DenseGCNConv):
                m.register_forward_hook(get_activation(f"{name}"))

    return model

from sklearn.metrics.cluster import completeness_score

class GCN(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_classes):
        super(GCN, self).__init__()

        self.conv0 = GCNConv(num_in_features, num_hidden_features)
        self.conv1 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv2 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv3 = GCNConv(num_hidden_features, num_hidden_features)

        # linear layers
        self.linear = nn.Linear(num_hidden_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=-1)



def run_experiment(seed, fold):
    epochs = 2000
    lr = 0.001
    batch_size = 32
    num_hidden_units = 10
    num_classes = 7
    K = 12

    # load data
    data = load_ba_shapes_node_class()

    x = data["x"]
    edges = data['edges']
    edges_t = data['edge_list'].numpy()
    y = data["y"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]

    # model training
    model = GCN(data["x"].shape[1], num_hidden_units, num_classes)

    # register hooks to track activation
    model = ba_shapes_model_utils.register_hooks(model)

    # train
    train_acc, test_acc, train_loss, test_loss = ba_shapes_model_utils.train(model, data, epochs, lr, if_interpretable_model=False)
    print("Testing Acc: ", test_acc[-1])

    # get model activations for complete dataset
    pred = model(x, edges)

    activation = torch.squeeze(ba_shapes_model_utils.activation_list['conv3']).detach().numpy()

    # find centroids
    kmeans_model = KMeans(n_clusters=K, random_state=seed)
    kmeans_model = kmeans_model.fit(activation[train_mask])
    used_centroid_labels = kmeans_model.predict(activation)
    centroid_labels = np.sort(np.unique(used_centroid_labels))
    centroids = kmeans_model.cluster_centers_


    # clustering efficency
    completeness = completeness_score(y, used_centroid_labels)
    print("Completeness: ", completeness)

    # # plot clustering
    # ba_shapes_model_utils.plot_clustering(seed, activation, y, centroids, centroid_labels, used_centroid_labels)
    # plt.close()

    # save concept scores - 3 colours plus 4 structures
    number_of_concepts = 7
    concept_scores = []
    for l in used_centroid_labels:
        concept_vector = np.zeros(number_of_concepts)
        res_sorted = kmeans_model.transform(activation)
        distances = res_sorted[:, l]
        node_idx = np.argsort(distances)[::][0]
        colour = x[node_idx][0]
        structure = y[node_idx]

        concept_vector[structure] = 1

        if colour == 1:
            concept_vector[4] = 1
        elif colour == 24:
            concept_vector[5] = 1
        elif colour == 80:
            concept_vector[6] = 1

        concept_scores.append(concept_vector)

    # print(concept_scores)
    concept_scores = np.array(concept_scores)
    # print(concept_scores.shape)


    # save embedding encoding
    embedding_encoding = []
    for concept_vector in concept_scores:
        embedding_vector = []
        for i, c in enumerate(concept_vector):
            if c == 1:
                embedding_vector.append(np.array(centroids[i]))
            else:
                embedding_vector.append(np.zeros(centroids[0].shape))

        embedding_encoding.append(embedding_vector)

    # print(embedding_encoding)
    embedding_encoding = np.array(embedding_encoding)
    # print(embedding_encoding.shape)

    # train_mask
    # test_mask
    path = f'./results/ba_shapes/{fold}'
    torch.save(model.state_dict(), f'{path}/model.pt')
    np.save(f'{path}/embedding_encoding.npy', embedding_encoding)
    np.save(f'{path}/concept_scores.npy', concept_scores)
    np.save(f'{path}/train_mask.npy', train_mask)
    np.save(f'{path}/test_mask.npy', test_mask)
    np.save(f'{path}/activations.npy', activation)
    np.save(f'{path}/y.npy', y)


def main():
    # run multiple times for confidence interval - seeds generated using Google's random number generator
    random_seeds = [42, 19, 76, 58, 92]

    for fold, seed in enumerate(random_seeds):
        print("\nSTART EXPERIMENT-----------------------------------------\n")
        seed_everything(seed)
        run_experiment(seed, fold)
        print("\nEND EXPERIMENT-------------------------------------------\n")

if __name__ == '__main__':
    main()
