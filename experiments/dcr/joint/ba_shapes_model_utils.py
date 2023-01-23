import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DenseGCNConv, dense_diff_pool, global_mean_pool
import torch.nn.functional as F
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import seaborn as sns
from matplotlib import rc


from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
import torch

global activation_list
activation_list = {}

# from dcr.data.ba_shapes import NUMBER_OF_CLASSES
NUMBER_OF_CLASSES = 2

def get_activation(idx):
    '''Learned from: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6'''
    def hook(model, in_put, output):
        if idx == "diff_pool":
            ret_labels = ["pooled_node_feat_matrix", "coarse_adj", "link_pred_loss", "entropy_reg"]
            for l, t in zip(ret_labels, output):
                activation_list[f"{idx}_{l}"] = t.detach()
        else:
            activation_list[idx] = output.detach()
    return hook


def register_hooks(model):
    for name, m in model.named_modules():
            if isinstance(m, GCNConv) or isinstance(m, nn.Linear) or isinstance(m, DenseGCNConv):
                m.register_forward_hook(get_activation(f"{name}"))

    return model


def reshape_graph_to_node_data(graph_data, batch):
    node_data = []

    i = 0
    for val in batch:
        if i == val:
            if torch.is_tensor(graph_data[i]):
                node_data.append(graph_data[i].detach().numpy())
            else:
                node_data.append(graph_data[i])
        else:
            i += 1
            if torch.is_tensor(graph_data[i]):
                node_data.append(graph_data[i].detach().numpy())
            else:
                node_data.append(graph_data[i])

    return torch.from_numpy(np.array(node_data))

def squared_dist(A, B):
    row_norms_A = torch.sum(torch.square(A), dim=1)
    row_norms_A = torch.reshape(row_norms_A, (-1, 1))

    row_norms_B = torch.sum(torch.square(B), dim=1)
    row_norms_B = torch.reshape(row_norms_B, (1, -1))

    return row_norms_A - 2 * torch.matmul(A, torch.transpose(B, 0, 1)) + row_norms_B


def quantization(input, output):
    D = squared_dist(input, torch.transpose(output, 0, 1))
    d_min = torch.max(D, dim=-1)[0]
    Q = torch.norm(d_min)

    return Q


def weights_init(m):
    # if isinstance(m, GCNConv):
    #     torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
    #     torch.nn.init.uniform_(m.bias.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("tanh"))
        torch.nn.init.uniform_(m.bias.data)


def test(model, node_data_x, node_data_y, edge_list, mask, if_interpretable_model=True):
    # enter evaluation mode
    model.eval()
    correct = 0

    if if_interpretable_model:
        _, pred = model(node_data_x, edge_list)
    else:
        pred = model(node_data_x, edge_list)
    pred = pred.max(dim=1)[1]

    correct += pred[mask].eq(node_data_y[mask]).sum().item()
    return correct / (len(node_data_y[mask]))


def create_embedding_mask(row, col):
    cutoff = int(col / 2)
    x_mask = np.zeros((row, col), dtype=bool)
    x_mask[:, :cutoff] = 1
    adj_mask = ~x_mask

    return x_mask, adj_mask

def train(model, data, epochs, lr, if_interpretable_model=True, modified_loss=False, stacked_loss=False):
    # get data
    x = data["x"]
    edges = data["edges"].long()
    y = data["y"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # list of accuracies
    train_accuracies, test_accuracies, train_losses, test_losses = list(), list(), list(), list()

    if modified_loss or stacked_loss:
        x_mask, adj_mask = create_embedding_mask(x.shape[0], model.cluster_encoding_size)

    # iterate for number of epochs
    for epoch in range(epochs):
            # set mode to training
            model.train()

            # input data
            optimizer.zero_grad()

            if if_interpretable_model:
                concepts, out = model(x, edges)
            else:
                out = model(x, edges)

            # calculate loss
    #         quantization_loss = model_utils.quantization(model.gnn_embedding[train_mask], model.cluster_centroids)
    #         loss = F.cross_entropy(out[train_mask], y[train_mask]) + 0.01 * te.nn.functional.entropy_logic_loss(model) + quantization_loss
            if modified_loss:
                i = torch.eye(edges.shape[0])
                x_concepts, x_out = model(x, i, x_mask)

                ones = torch.ones(x.shape)
                adj_concepts, adj_out = model(ones, edges, adj_mask)

                loss = F.cross_entropy(out[train_mask], y[train_mask]) + 0.1 * F.cross_entropy(x_out[train_mask], y[train_mask]) + 0.1 * F.cross_entropy(adj_out[train_mask], y[train_mask])
            elif stacked_loss:
                i = torch.eye(edges.shape[0])
                x_concepts, x_out = model(x, i, x_mask)

                ones = torch.ones(x.shape)
                adj_concepts, adj_out = model(ones, edges, adj_mask)

                stacked_out = torch.vstack((x_out[train_mask], adj_out[train_mask], out[train_mask]))
                stacked_y = torch.cat((y[train_mask], y[train_mask], y[train_mask]), 0)
                loss = F.cross_entropy(stacked_out, stacked_y)
            else:
                loss = F.cross_entropy(out[train_mask], y[train_mask]) #+ 1e-6 * quantization_loss
            # else:
            #     # embedding_diff = torch.max(torch.subtract(model.exa, torch.add(model.ex, model.ea)), dim=-1)[0]
            #     quantization_loss = quantization(model.exa[train_mask], torch.transpose(torch.add(model.ex[train_mask], model.ea[train_mask]), 0, 1))
            #     loss = F.cross_entropy(out[train_mask], y[train_mask]) + 1e-6 * quantization_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                test_loss = F.cross_entropy(out[test_mask], y[test_mask])
                train_acc = test(model, x, y, edges, train_mask, if_interpretable_model=if_interpretable_model)
                test_acc = test(model, x, y, edges, test_mask, if_interpretable_model=if_interpretable_model)

            ## add to list and print
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())

            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, loss.item(), train_acc, test_acc), end = "\r")

    model.eval()
    return train_accuracies, test_accuracies, train_losses, test_losses


def test_graph_class(model, dataloader, if_interpretable_model=True):
    # enter evaluation mode
    correct = 0
    for data in dataloader:
        if if_interpretable_model:
            concepts, out = model(data.x, data.edge_index, data.batch)
        else:
            out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(dataloader.dataset)


def train_graph_class(model, train_loader, test_loader, full_loader, epochs, lr, if_interpretable_model=True):
    # register hooks to track activation
    model = register_hooks(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # list of accuracies
    train_accuracies, test_accuracies, train_loss, test_loss = list(), list(), list(), list()

    for epoch in range(epochs):
        model.train()

        running_loss = 0
        num_batches = 0
        for data in train_loader:
            model.train()

            optimizer.zero_grad()

            if if_interpretable_model:
                concepts, out = model(data.x, data.edge_index, data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)

            # calculate loss
            one_hot = torch.nn.functional.one_hot(data.y, num_classes=NUMBER_OF_CLASSES).type_as(out)
            if out.shape[1] == 1 or one_hot.shape[1] == 1:
                print("What ", out.shape)
                print(out)
                print("What2 ", data.y.shape, " ", one_hot.shape)
                print(data.y)
                print(one_hot)
            loss = criterion(out, one_hot)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            running_loss += loss.item()
            num_batches += 1

            optimizer.step()

        # get accuracy
        train_acc = test_graph_class(model, train_loader, if_interpretable_model=if_interpretable_model)
        test_acc = test_graph_class(model, test_loader, if_interpretable_model=if_interpretable_model)

        # add to list and print
        model.eval()
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # get testing loss
        test_running_loss = 0
        test_num_batches = 0
        for data in test_loader:
            if if_interpretable_model:
                concepts, out = model(data.x, data.edge_index, data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)

            one_hot = torch.nn.functional.one_hot(data.y, num_classes=NUMBER_OF_CLASSES).type_as(out)
            if out.shape[1] == 1 or one_hot.shape[1] == 1:
                print("What ", out.shape)
                print(out)
                print("What2 ", data.y.shape, " ", one_hot.shape)
                print(data.y)
                print(one_hot)
            test_running_loss += criterion(out, one_hot).item()
            test_num_batches += 1

        train_loss.append(running_loss / num_batches)
        test_loss.append(test_running_loss / test_num_batches)

        print('Epoch: {:03d}, Train Loss: {:.5f}, Test Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, train_loss[-1], test_loss[-1], train_acc, test_acc), end = "\r")

    return train_accuracies, test_accuracies, train_loss, test_loss

class Pool(torch.nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x, batch):
        x = global_mean_pool(x, batch)
        return x


class DiffPool(torch.nn.Module):
    def __init__(self):
        super(DiffPool, self).__init__()

    def forward(self, x, adj, s):
        return dense_diff_pool(x, adj, s)



def plot_clustering(seed, activation, y, centroids, centroid_labels, used_centroid_labels, id_title="Node ", id_path="node"):
    all_data = np.vstack([activation, centroids])

    tsne_model = TSNE(n_components=2, random_state=seed)
    all_data2d = tsne_model.fit_transform(all_data)

    d = all_data2d[:len(activation)]
    centroids2d = all_data2d[len(activation):]

    fig = plt.figure(figsize=[15, 5])
    fig.suptitle(f"{id_title}Clustering of Activations in Last Layer of GCN")

    ax = plt.subplot(1, 3, 1)
    ax.set_title("Real Labels")
    p = sns.color_palette("husl", len(np.unique(y)))
    sns.scatterplot(x=d[:, 0], y=d[:, 1], hue=y, palette=p)

    ax = plt.subplot(1, 3, 2)
    ax.set_title("Model's Clusters")
    p = sns.color_palette("husl", len(np.unique(used_centroid_labels)))
    sns.scatterplot(x=d[:, 0], y=d[:, 1], hue=used_centroid_labels, palette=p, legend=None)

    ax = plt.subplot(1, 3, 3)
    ax.set_title("Model's Cluster Centroids")
    p = sns.color_palette("husl", len(np.unique(used_centroid_labels)))
    sns.scatterplot(x=d[:, 0], y=d[:, 1], hue=used_centroid_labels, palette=p, legend=None, alpha=0.3)

    p = sns.color_palette("husl", len(centroids))
    sns.scatterplot(x=centroids2d[:, 0], y=centroids2d[:, 1], hue=list(range(len(centroids))), palette=p, alpha=0.7,
                    legend=None, **{'s': 600, 'marker': '*', 'edgecolors': None})

    plt.show()
