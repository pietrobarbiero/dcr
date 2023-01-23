import torch
import numpy as np
import networkx as nx
import os

from torch.utils.data import Subset
from torch_geometric.data import Data
import random

NUMBER_OF_CONCEPTS = 5
NUMBER_OF_CLASSES = 6
NUMBER_OF_EXAMPLES = 1500

def load_graph(idx):
    G = nx.read_gpickle(f"./data/ba_shapes/{idx}_graph_ba_20_1.gpickel")
    node_labels = np.load(f"./data/ba_shapes/{idx}_role_ids_ba_20_1.npy")
    concept_labels = np.load(f"./data/ba_shapes/{idx}_color_concepts_ba_20_1.npy")
    graph_label = np.load(f"data/ba_shapes/{idx}_y_ba_20_1.npy")

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]

    features = torch.from_numpy(features).float()
    graph_label = torch.from_numpy(graph_label)

    # print(np.unique(graph_label))
    # print(NUMBER_OF_CLASSES)
    # one_hot_labels = torch.nn.functional.one_hot(graph_label, num_classes=NUMBER_OF_CLASSES)

    edge_list = torch.from_numpy(np.array(G.edges))
    edges = edge_list.transpose(0, 1)

    encoded_concepts = []
    for c in concept_labels:
        # print(f"This {l} and {c}")
        encoded_c = np.zeros(NUMBER_OF_CONCEPTS)
        # encoded_c[l] = 1
        encoded_c[int(c)] = 1

        encoded_concepts.append(encoded_c)

    encoded_concepts = torch.from_numpy(np.array(encoded_concepts))

    data = Data(x=features, edge_index=edges, y=graph_label, encoded_concepts=encoded_concepts)

    return data


def load_ba_shapes():
    graphs = [load_graph(idx) for idx in range(NUMBER_OF_EXAMPLES)]
    random.shuffle(graphs)

    train_idx = sorted(np.random.choice(np.arange(len(graphs)), size=int(0.8*len(graphs)), replace=False))
    test_idx = np.setdiff1d(np.arange(len(graphs)), train_idx)
    train_data = Subset(graphs, train_idx)
    test_data = Subset(graphs, test_idx)

    in_concepts = NUMBER_OF_CONCEPTS
    out_concepts = NUMBER_OF_CLASSES
    concept_names = ['BA', 'Middle', 'Bottom', 'Top', 'Circle','Green', 'Red', 'Blue']
    class_names = ['BA', 'Middle', 'Bottom', 'Top', 'Circle']
    return graphs, train_data, test_data, in_concepts, out_concepts, concept_names, class_names

def load_ba_shapes_node_class():
    train_split = 0.8
    if_adj = False

    G = nx.readwrite.read_gpickle(f"./data/ba_shapes/graph_ba_300_80.gpickel")
    role_ids = np.load(f"./data/ba_shapes/role_ids_ba_300_80.npy")
    concepts = np.load(f"./data/ba_shapes/concepts_ba_300_80.npy")

    # relabel
    labels = []

    print(concepts)
    print(role_ids)
    for l, c in zip(role_ids, concepts):
        offset = 3
        if l == 0 and c == 1:
            labels.append(l)

        if c == 2 or c == 15:
            labels.append(l)
        # elif c == 24:
        #     if l == 4:
        #         print(l + offset)
        #     labels.append(l + offset)
        elif c == 80:
            labels.append(l + offset)

    labels = np.array(labels)

    print(len(role_ids))
    print(len(labels))
    print(len(concepts))
    assert(len(role_ids) == len(labels))


    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels)

    edge_list = torch.from_numpy(np.array(G.edges))

    edges = edge_list.transpose(0, 1)
    if if_adj:
        edges = torch.tensor(nx.to_numpy_matrix(G), requires_grad=True, dtype=torch.float)

    train_mask = np.random.rand(len(features)) < train_split
    test_mask = ~train_mask

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(labels))
    print("Number of classes: ", len(set(labels)))
    print("Number of edges: ", len(edges))



    data = {"x": features, "y": labels, "edges": edges, "edge_list": edge_list, "train_mask": train_mask, "test_mask": test_mask}

    return data
