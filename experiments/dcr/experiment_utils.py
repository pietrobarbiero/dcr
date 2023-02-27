import joblib
import numpy as np
import os
import torch
from pathlib import Path
import pytorch_lightning as pl

import cem.train.intervention_utils as intervention_utils


def dump_end_embeddings(
    x_train,
    x_test,
    model,
    activations_dir,
    n_concepts,
    model_name,
    split=0,
    x_val=None,
    gpu=1,
    batch_size=512,
):
    Path(activations_dir).mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(
        gpus=gpu,
    )
    for name, data in [
        ('train', x_train),
        ('test', x_test),
        ('val', x_val),
    ]:
        if data is None:
            continue
        batch_results = trainer.predict(
            model,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.FloatTensor(data)),
                batch_size=batch_size,
            )
        )
        c_sem = np.concatenate(
            list(map(lambda x: x[0], batch_results)),
            axis=0,
        )
        c_pred = np.concatenate(
            list(map(lambda x: x[1], batch_results)),
            axis=0,
        )
        if len(c_pred.shape) == 2:
            (n, dim) = c_pred.shape
            c_pred = np.reshape(c_pred, [n, n_concepts, dim//n_concepts])
        y_pred = np.concatenate(
            list(map(lambda x: x[2], batch_results)),
            axis=0,
        )
        np.save(
            os.path.join(
                activations_dir,
                f"{name}_c_embeddings_{model_name}_{split}.npy",
            ),
            c_pred,
        )
        np.save(
            os.path.join(
                activations_dir,
                f"{name}_c_pred_semantics_{model_name}_{split}.npy",
            ),
            c_sem,
        )
        np.save(
            os.path.join(
                activations_dir,
                f"{name}_y_pred_{model_name}_{split}.npy",
            ),
            y_pred,
        )

def dumb_pos_neg_cem_embs(
    config,
    x_train,
    x_test,
    activations_dir,
    n_concepts,
    model_name,
    result_dir,
    n_tasks,
    imbalance=None,
    split=0,
    x_val=None,
    gpu=1,
    batch_size=512,
):
    Path(activations_dir).mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(
        gpus=gpu,
    )
    config["shared_prob_gen"] = config.get("shared_prob_gen", False)
    config["per_concept_weight"] = config.get(
        "per_concept_weight",
        False,
    )
    model = intervention_utils.load_trained_model(
        config=config,
        n_tasks=n_tasks,
        n_concepts=n_concepts,
        result_dir=result_dir,
        split=split,
        imbalance=imbalance,
        intervention_idxs=list(range(n_concepts)),
    )
    for name, data in [
        ('train', x_train),
        ('test', x_test),
        ('val', x_val),
    ]:
        if data is None:
            continue
        # First let's get all the positive embeddings by tricking the
        # model into thinking that we are intervening in all concepts
        # and all ground truth labels are 1
        pos_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(data),
                torch.FloatTensor(
                    np.ones((data.shape[0], n_concepts)),
                ),
            ),
            batch_size=batch_size,
        )
        batch_results = trainer.predict(model, pos_data_loader)
        c_pred = np.concatenate(
            list(map(lambda x: x[1], batch_results)),
            axis=0,
        )
        np.save(
            os.path.join(
                activations_dir,
                f"{name}_c_pos_embeddings_{model_name}_{split}.npy",
            ),
            c_pred,
        )
        
        # Now the complement trick to get the negative embeddings
        neg_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(data),
                torch.FloatTensor(
                    np.zeros((data.shape[0], n_concepts)),
                ),
            ),
            batch_size=batch_size,
        )
        batch_results = trainer.predict(model, neg_data_loader)
        c_pred = np.concatenate(
            list(map(lambda x: x[1], batch_results)),
            axis=0,
        )
        np.save(
            os.path.join(
                activations_dir,
                f"{name}_c_neg_embeddings_{model_name}_{split}.npy",
            ),
            c_pred,
        )