from  torch.utils.data import TensorDataset

from model import DeepConceptReasoner, pretty_print_explain
from datasets import create_single_digit_addition, addition_dataset
from dcr.semantics import ProductTNorm, GodelTNorm
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torchvision
import torch
import os
import random
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def load_data(number_digits, size = None, device = "cpu"):


    concept_names, explanations = create_single_digit_addition(number_digits)

    X, y, z = addition_dataset(True, number_digits, size=size, device = device)
    X_test, y_test, z_test = addition_dataset(False, number_digits, device = device)

    train_dataset = TensorDataset(*X, *y, z)
    test_dataset = TensorDataset(*X_test, *y_test, z_test)

    # cl, cnt = torch.unique(y, return_counts=True)
    # weights = {int(c): int(cc) for c,cc in zip(cl,cnt)}

    return train_dataset, test_dataset, concept_names, explanations #, weights


def main(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    epochs = 1000
    learning_rate = 0.001
    limit_batches = 1.0
    emb_size = 30
    number_digits = 2
    num_classes = 10
    number_additions = 19
    temperature = 1000000 # We do not want filters as there are mutually exclusive atoms. We want the network to learn them.
    temperature_gumbel = 1.75
    logic = GodelTNorm()
    size = 30000
    num_concepts_supervisions = 10
    # batch_size = size / num_concepts_supervisions
    # assert batch_size >= 1
    # assert batch_size.is_integer()
    # batch_size = int(batch_size)
    # num_concepts_supervisions = int(num_concepts_supervisions / (size // batch_size))
    batch_size = 256
    device = 'cuda'
    # device = 'cpu'



    results_dir = f"./results_test_175/seed%d/" % seed
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'model.pt')

    train_data, test_data, concept_names, explanations = load_data(number_digits, size = size * 2, device = device)




    train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, len(test_data), shuffle=False)




    to_explain = addition_dataset(True, number_digits, device=device)
    to_explain = to_explain[0][0][:10000], to_explain[0][1][:10000],


    model = DeepConceptReasoner(emb_size=emb_size,
                                num_concepts = num_classes,
                                num_digits = number_digits,
                                num_additions=number_additions,
                                learning_rate=learning_rate,
                                logic=logic,
                                temperature=temperature,
                                temperature_gumbel=temperature_gumbel,
                                num_concepts_supervisions=num_concepts_supervisions,
                                to_explain = to_explain,
                                class_weights = None)


    # if not os.path.exists(model_path):
    print(f'Running epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}')
    logger = TensorBoardLogger(save_dir=results_dir, name="lightning_logs")
    csv_logger = CSVLogger(save_dir=results_dir, name="lightning_lcsv_ogs")



    trainer = pl.Trainer(max_epochs=300,
                         accelerator="gpu",
                         # check_val_every_n_epoch=100,
                         logger=[logger ,csv_logger])
    trainer.fit(model=model, train_dataloaders=train_dl)
    res = trainer.validate(model=model, dataloaders=[test_dl])




    # torch.save(model.state_dict(), model_path)
    # model.load_state_dict(torch.load(model_path))


    if device == "cuda":
        model = model.cuda()



    #Explain
    explanations = sorted(model.explain(to_explain), key=lambda x: int(x["class"].split("_")[1]))
    # explanations_filtered = [e for e in explanations if e["class"] in ("y_0", "y_1")]

    with open(os.path.join(results_dir,'res.txt'), 'w') as f:
        print(res, file=f)
        pretty_print_explain(explanations, file=f)


    train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)


    for dl, split in zip([train_dl, test_dl], ["train", "test"]):
        Y_pred = []
        Y_emb = []
        Z_pred = []
        Z_n_pred = []
        Z = []
        for I in dl:
            X = I[:2]
            z = I[4]
            y_pred, y_emb, z_pred, z_n_pred = model.forward(X)
            Z.append(z)
            Z_pred.append(z_pred)
            Z_n_pred.append(z_n_pred)
            Y_pred.append(y_pred)
            Y_emb.append(y_emb)

        Z = torch.concat(Z, dim=0)
        Z_pred = torch.concat(Z_pred, dim=0)
        Z_n_pred = torch.concat(Z_n_pred, dim=0)
        Y_pred = torch.concat(Y_pred, dim=0)
        Y_emb = torch.concat(Y_emb, dim=0)


        torch.save(Z, os.path.join(results_dir,"%s_task_labels.pt" % split))
        torch.save(Z_pred, os.path.join(results_dir,"%s_task_predictions_dcr.pt" % split))
        torch.save(Z_n_pred, os.path.join(results_dir,"%s_task_predictions_nn.pt" % split))
        torch.save(Y_pred, os.path.join(results_dir,"%s_concept_predictions.pt" % split))
        torch.save(Y_emb, os.path.join(results_dir,"%s_concept_embeddings.pt" % split))



if __name__ == '__main__':
    for seed in range(5):
        main(seed)
