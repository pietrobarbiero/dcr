import torch
import numpy as np
import networkx as nx

def load_ba_shapes():
    G = nx.read_gpickle(f"./data/ba_shapes/graph_ba_300_80.gpickel")
    labels = np.load(f"./data/ba_shapes/role_ids_ba_300_80.npy")
    concept_labels = np.load(f"./data/ba_shapes/color_concepts_ba_300_80.npy")

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]


    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels)
    labels = torch.nn.functional.one_hot(labels)

    concept_labels = torch.from_numpy(concept_labels)

    edge_list = torch.from_numpy(np.array(G.edges))
    edges = edge_list.transpose(0, 1)
    # if if_adj:
    #     edges = torch.tensor(nx.to_numpy_matrix(G), requires_grad=True, dtype=torch.float)

    train_split = 0.8
    train_mask = np.random.rand(len(features)) < train_split
    test_mask = ~train_mask

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(labels))
    print("Number of classes: ", len(set(labels)))
    print("Number of edges: ", len(edges))

    encoded_concepts = []
    for l, c in zip(labels, concept_labels):
        encoded_c = np.zeros(7)
        encoded_c[l] = 1
        encoded_c[c] = 1

        encoded_concepts.append(encoded_c)

    encoded_concepts = torch.from_numpy(np.array(encoded_concepts))

    data = {"x": features, "y": labels, "concept_labels": encoded_concepts, "edges": edges, "edge_list": edge_list, "train_mask": train_mask, "test_mask": test_mask}

    in_concepts = 8
    out_concepts = 5
    concept_names = ['BA', 'Middle', 'Bottom', 'Top', 'Circle','Green', 'Red', 'Blue']
    class_names = ['BA', 'Middle', 'Bottom', 'Top', 'Circle']

    return data, in_concepts, out_concepts, concept_names, class_names
