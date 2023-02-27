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

from dcr.data.ba_shapes import load_ba_shapes, NUMBER_OF_CLASSES

from sklearn.cluster import KMeans

import ba_shapes_model_utils
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

global activation_list
activation_list = {}

from sklearn.metrics.cluster import completeness_score

from sklearn.tree import DecisionTreeClassifier

# model definition
class GCN(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_classes):
        super(GCN, self).__init__()

        self.conv0 = GCNConv(num_in_features, num_hidden_features)
        self.conv1 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv2 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv3 = GCNConv(num_hidden_features, 10)

        self.pool = ba_shapes_model_utils.Pool()

        # linear layers
        self.linear = nn.Linear(10, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv0(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)

        self.gnn_node_embedding = x

        x = self.pool(x, batch)

        self.gnn_graph_embedding = x

        x = self.linear(x)

        return F.log_softmax(x, dim=-1)

def load_mutagenicity(batch_size):
    graphs = TUDataset(root='../data/', name='REDDIT-BINARY', transform=torch_geometric.transforms.Constant())
    graphs = graphs.shuffle()

    train_split = 0.8
    train_idx = int(len(graphs) * train_split)
    train_set = graphs[:train_idx]
    test_set = graphs[train_idx:]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    full_train_loader = DataLoader(train_set, batch_size=int(len(train_set) * 0.1), shuffle=False)
    full_test_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1), shuffle=False)

    full_loader = DataLoader(test_set, batch_size=int(len(test_set)), shuffle=False)
    small_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1), shuffle=False)

    train_zeros = 0
    train_ones = 0

    for data in train_set:
        train_ones += np.sum(data.y.detach().numpy())
        train_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    test_zeros = 0
    test_ones = 0

    for data in test_set:
        test_ones += np.sum(data.y.detach().numpy())
        test_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    return graphs, train_loader, test_loader, full_train_loader, full_test_loader, full_loader, small_loader


def run_experiment(seed, fold):
    epochs = 1000
    lr = 0.001
    batch_size = 16
    num_hidden_units = 40
    num_classes = 2
    K = 30

    # load data
    graphs, train_dl, test_dl, full_train_dl, full_test_dl, full_loader, small_loader = load_mutagenicity(batch_size)

    # model training
    model = GCN(graphs.num_node_features, num_hidden_units, graphs.num_classes)

    # register hooks to track activation
    model = ba_shapes_model_utils.register_hooks(model)

    # train
    train_acc, test_acc, train_loss, test_loss = ba_shapes_model_utils.train_graph_class(model, train_dl, test_dl, full_loader, epochs, lr, if_interpretable_model=False)
    print(test_acc[-1])

    # get model activations for complete dataset
    train_data = next(iter(full_train_dl))
    _ = model(train_data.x, train_data.edge_index, train_data.batch)
    train_activation = model.gnn_node_embedding

    test_data = next(iter(full_test_dl))
    _ = model(test_data.x, test_data.edge_index, test_data.batch)
    test_activation = model.gnn_node_embedding

    activation = torch.vstack((train_activation, test_activation)).detach().numpy()

    y = torch.cat((train_data.y, test_data.y))
    expanded_train_y = ba_shapes_model_utils.reshape_graph_to_node_data(train_data.y, train_data.batch)
    expanded_test_y = ba_shapes_model_utils.reshape_graph_to_node_data(test_data.y, test_data.batch)
    expanded_y = torch.cat((expanded_train_y, expanded_test_y))

    train_mask = np.zeros(activation.shape[0], dtype=bool)
    train_mask[:train_activation.shape[0]] = True
    test_mask = ~train_mask

    # find centroids
    kmeans_model = KMeans(n_clusters=K, random_state=seed)
    kmeans_model = kmeans_model.fit(activation[train_mask])
    used_centroid_labels = kmeans_model.predict(activation)
    centroid_labels = np.sort(np.unique(used_centroid_labels))
    centroids = kmeans_model.cluster_centers_

    # completeness = completeness_score(expanded_y, used_centroid_labels)

    clf = DecisionTreeClassifier(random_state=seed)
    clf = clf.fit(used_centroid_labels[train_mask].reshape(-1, 1), expanded_y[train_mask])
    completeness = clf.score(used_centroid_labels[test_mask].reshape(-1, 1), expanded_y[test_mask])
    print("Completeness score ", completeness)

    ba_shapes_model_utils.plot_clustering(seed, test_activation.detach().numpy(), expanded_test_y, centroids, centroid_labels, used_centroid_labels[test_mask])

    offset = train_data.batch[-1] + 1
    batch = torch.cat((train_data.batch, test_data.batch + offset))

    # save concept scores - 3 colours plus 4 structures
    number_of_concepts = K
    concept_scores = []

    current_counter = batch[0]
    current_concept_vector = np.zeros(number_of_concepts)
    for i in batch:
        if i != current_counter:
            concept_scores.append(current_concept_vector)
            current_counter = i
            current_concept_vector = np.zeros(number_of_concepts)

        current_concept_vector[used_centroid_labels[i]] = 1

    concept_scores.append(current_concept_vector)
    concept_scores = np.array(concept_scores)

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

    embedding_encoding = np.array(embedding_encoding)

    path = f'./results/reddit/{fold}'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), f'{path}/model.pt')
    np.save(f'{path}/embedding_encoding.npy', embedding_encoding)
    np.save(f'{path}/concept_scores.npy', concept_scores)
    np.save(f'{path}/train_mask.npy', train_mask)
    np.save(f'{path}/test_mask.npy', test_mask)
    np.save(f'{path}/activations.npy', activation)
    np.save(f'{path}/y_node.npy', expanded_y)
    np.save(f'{path}/y_graph.npy', y)

def main():
    # run multiple times for confidence interval - seeds generated using Google's random number generator
    random_seeds = [42, 19, 76, 58, 92]

    for fold, seed in enumerate(random_seeds):
        print("\nSTART EXPERIMENT-----------------------------------------\n")
        seed_everything(seed)
        run_experiment(seed, fold)
        print("\nEND EXPERIMENT-------------------------------------------\n")
        break


if __name__ == '__main__':
    main()
