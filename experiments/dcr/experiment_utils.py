import joblib
import numpy as np
import os
import torch
from pathlib import Path
import pytorch_lightning as pl

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