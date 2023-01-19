import numpy as np
import torch
import tensorflow as tf

################################################################################
## DATASET GENERATORS
################################################################################

def generate_xor_data(size, one_hot=True):
    # sample from normal distribution
    x = np.random.uniform(0, 1, (size, 2))
    c = np.stack([
        x[:, 0] > 0.5,
        x[:, 1] > 0.5,
    ]).T
    y = np.logical_xor(c[:, 0], c[:, 1])

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    if one_hot:
        y = torch.FloatTensor(tf.one_hot(y, len(np.unique(y))).numpy())
    y = torch.FloatTensor(y)
    return x, y, c


def generate_trig_data(size, one_hot=True):
    h = np.random.normal(0, 2, (size, 3))
    x, y, z = h[:, 0], h[:, 1], h[:, 2]

    # raw features
    input_features = np.stack([
        np.sin(x) + x,
        np.cos(x) + x,
        np.sin(y) + y,
        np.cos(y) + y,
        np.sin(z) + z,
        np.cos(z) + z,
        x ** 2 + y ** 2 + z ** 2,
    ]).T

    # concetps
    concetps = np.stack([
        x > 0,
        y > 0,
        z > 0,
    ]).T

    # task
    downstream_task = (x + y + z) > 1

    input_features = torch.FloatTensor(input_features)
    concetps = torch.FloatTensor(concetps)
    if one_hot:
        downstream_task = torch.FloatTensor(
            tf.one_hot(y, len(np.unique(downstream_task))).numpy()
        )
    else:
        downstream_task = torch.FloatTensor(y)
    return input_features, downstream_task, concetps


def generate_dot_data(size, one_hot=True):
    # sample from normal distribution
    emb_size = 2
    v1 = np.random.randn(size, emb_size) * 2
    v2 = np.ones(emb_size)
    v3 = np.random.randn(size, emb_size) * 2
    v4 = -np.ones(emb_size)
    x = np.hstack([v1+v3, v1-v3])
    c = np.stack([
        np.dot(v1, v2).ravel() > 0,
        np.dot(v3, v4).ravel() > 0,
    ]).T
    y = ((v1*v3).sum(axis=-1) > 0).astype(np.int64)

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    if one_hot:
        y = torch.FloatTensor(tf.one_hot(y, len(np.unique(y))).numpy())
    y = torch.FloatTensor(y)
    return x, y, c


################################################################################
## GROUPED DATASET GENERATOR
################################################################################

def generate_synth_dataset(config, dataset):
    if dataset == "xor":
        generate_data = generate_xor_data
    elif dataset in ["trig", "trigonometry"]:
        generate_data = generate_trig_data
    elif dataset in ["vector", "dot"]:
        generate_data = generate_dot_data
    else:
        raise ValueError(f"Unsupported dataset {dataset}")
    dataset_size = config['dataset_size']
    batch_size = config["batch_size"]
    x, y, c = generate_data(int(dataset_size * 0.7))
    train_data = torch.utils.data.TensorDataset(x, y, c)
    train_dl = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', -1),
    )

    x_test, y_test, c_test = generate_data(int(dataset_size * 0.2))
    test_data = torch.utils.data.TensorDataset(x_test, y_test, c_test)
    test_dl = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=config.get('num_workers', -1),
    )

    x_val, y_val, c_val = generate_data(int(dataset_size * 0.1))
    val_data = torch.utils.data.TensorDataset(x_val, y_val, c_val)
    val_dl = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=config.get('num_workers', -1),
    )
    return train_dl, test_dl, val_dl