from torch.nn import BCELoss
import pytorch_lightning as pl
import torch
import os
import copy
import joblib
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


import sys
sys.path.append(".")

from dcr.data.celeba import load_celeba
from dcr.models import DeepConceptReasoner
from dcr.semantics import GodelTNorm

def find_class_imbalance(c_train):
    imbalance_ratio = []
    n, n_attr = c_train.shape
    n_ones = np.sum(c_train, axis=0)
    total = np.array([n] * n_attr)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j]/n_ones[j] - 1)
    return imbalance_ratio

def random_int_policy(num_groups_intervened, concept_group_map):
    selected_groups_for_trial = np.random.choice(
        list(concept_group_map.keys()),
        size=num_groups_intervened,
        replace=False,
    )
    intervention_idxs = []
    for selected_group in selected_groups_for_trial:
        intervention_idxs.extend(concept_group_map[selected_group])
    return sorted(intervention_idxs)


def main():
    epochs = 50
    learning_rate = 0.0008
    emb_size = 16
    batch_size = 128
    limit_batches = 1.0
    intervention_trials = 3
    concept_selection_policy = random_int_policy
    num_workers = 6

    train_data, test_data, n_concepts, out_concepts, concept_names, class_names = load_celeba()
    train_dl = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_dl = DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    x_test, c_test, y_test = [], [], []
    for x, (c, y) in test_dl:
        x_test.append(x)
        c_test.append(c)
        y_test.append(y)
    
    x_train, c_train, y_train = [], [], []
    for x, (c, y) in train_dl:
        x_train.append(x)
        c_train.append(c)
        y_train.append(y)
    
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    c_train = np.concatenate(c_train, axis=0)
    print("x_train.shape =", x_train.shape)
    print("y_train.shape =", y_train.shape)
    print("y_train distribution =", np.mean(y_train, axis=0))
    print("c_train.shape =", c_train.shape)
    print("c_train distribution =", np.mean(c_train, axis=0))
    imbalance = find_class_imbalance(c_train)
    print("imbalance =", imbalance)
    
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    c_test = np.concatenate(c_test, axis=0)
    print("x_test.shape =", x_test.shape)
    print("y_test.shape =", y_test.shape)
    print("y_test distribution =", np.mean(y_test, axis=0))
    print("c_test.shape =", c_test.shape)
    print("c_test distribution =", np.mean(c_test, axis=0))
        
    concept_group_map = None
    
    for fold in range(5):
        results_dir = f"./results/celeba_ints/{fold}/"
        os.makedirs(results_dir, exist_ok=True)
        model_path = os.path.join(results_dir, 'model.pt')
        
        model_args = dict(
            in_concepts=n_concepts,
            out_concepts=out_concepts,
            emb_size=emb_size,
            concept_names=concept_names,
            class_names=class_names,
            learning_rate=learning_rate,
            loss_form=BCELoss(),
            loss_form_concept=BCELoss(weight=torch.FloatTensor(imbalance)),
            concept_loss_weight=1,
            class_loss_weight=10, #0.1,
            logic=GodelTNorm(),
            reasoner=True,
            temperature=1,
        )
        model = DeepConceptReasoner(**model_args)
        if not os.path.exists(model_path):
            print(f'Running epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}')
            logger = TensorBoardLogger(save_dir=results_dir, name="lightning_logs")
            trainer = pl.Trainer(
                max_epochs=epochs,
                devices=1,
                accelerator="gpu",
                limit_train_batches=limit_batches,
                limit_val_batches=limit_batches,
                logger=logger,
            )
            trainer.fit(model=model, train_dataloaders=train_dl)
            torch.save(model.state_dict(), model_path)

        model.load_state_dict(torch.load(model_path))
        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu",
            limit_test_batches=limit_batches,
        )
        [test_results] = trainer.test(model, dataloaders=test_dl)
        pred_results = trainer.predict(model, test_dl)
        y_pred = np.concatenate(
            list(map(lambda x: x[1], pred_results)),
            axis=0,
        )
        c_pred = np.concatenate(
            list(map(lambda x: x[0], pred_results)),
            axis=0,
        )
        test_results['auc'] = roc_auc_score(y_test, y_pred)
        test_results['acc'] = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        test_results['c_auc'] = 0
        test_results['c_acc'] = 0
        for i in range(n_concepts):
            test_results['c_auc'] += roc_auc_score(c_test[:, i], (c_pred[:, i] > 0.5).astype(np.int32))/n_concepts
            test_results['c_acc'] += accuracy_score(c_test[:, i], (c_pred[:, i] > 0.5).astype(np.int32))/n_concepts
        print(f"Test results! for fold {fold}:")
        for k, v in test_results.items():
            print(f'\t"{k}" -> {v*100:.2f}%')
        print("Performing interventions...")
        
        intervention_aucs = []
        intervention_accs = []
        # If no concept groups are given, then we assume that all concepts
        # represent a unitary group themselves
        concept_group_map = concept_group_map or dict(
            [(i, [i]) for i in range(n_concepts)]
        )
        groups = list(range(0, len(concept_group_map) + 1, 1))
        for j, num_groups_intervened in enumerate(groups):
            print(
                f"\tIntervening with {num_groups_intervened} out of "
                f"{len(concept_group_map)} concept groups"
            )
            auc_avg = []
            acc_avg = []
            for trial in range(intervention_trials):
                intervention_idxs = concept_selection_policy(
                    num_groups_intervened=num_groups_intervened,
                    concept_group_map=concept_group_map,
                )
                print(
                    f"\t\tFor trial {trial + 1}/{intervention_trials} for fold {fold} with "
                    f"{num_groups_intervened} groups intervened"
                )
                print(
                    f"\t\t\tThis results in concepts {intervention_idxs} being "
                    f"modified"
                )
                int_model_args = copy.deepcopy(model_args)
                int_model_args['intervention_idxs'] = intervention_idxs
                int_model = DeepConceptReasoner(**int_model_args)
                int_model.load_state_dict(torch.load(model_path))
                trainer = pl.Trainer(
                    gpus=1,
                )
                pred_results = trainer.predict(int_model, test_dl)
                y_pred = np.concatenate(
                    list(map(lambda x: x[1], pred_results)),
                    axis=0,
                )
                auc = roc_auc_score(y_test, y_pred)
                auc_avg.append(auc)
                acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
                acc_avg.append(acc)
                print(
                    f"\t\t\tFor model at fold {fold}, intervening with "
                    f"{num_groups_intervened} groups (trial {trial + 1}) gives "
                    f"test task AUC {auc * 100:.2f}% and accuracy {acc * 100:.2f}%."
                )
            print(
                f"\tTest AUC when intervening with {num_groups_intervened} "
                f"concept groups is "
                f"{np.mean(auc_avg) * 100:.2f}% ± {np.std(auc_avg)* 100:.2f}%."
            )
            print(
                f"\tTest accuracy when intervening with {num_groups_intervened} "
                f"concept groups is "
                f"{np.mean(acc_avg) * 100:.2f}% ± {np.std(acc_avg)* 100:.2f}%."
            )
            intervention_aucs.append(np.mean(auc_avg))
            intervention_accs.append(np.mean(acc_avg))
        test_results['intervention_aucs'] = intervention_aucs
        test_results['intervention_accs'] = intervention_accs
        joblib.dump(test_results, os.path.join(results_dir, 'test_results.joblib'))


if __name__ == '__main__':
    main()
