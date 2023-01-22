# In this file we perform the counterfactual experiments.
# We compare the the counterfactual explanations provided by:
#       - CBM (logistic regression over embedding score)
#       - ReluNN
#       - DCR
#       - MLP with Anchors explanations?

# In terms of L(x,x′,y′,λ)=λ⋅(^f(x′)−y′)2+d(x,x′),  d(x,x′)=∑^p_{j=1}(|x_j−x′_j|)/MAD_j
import copy

import numpy as np
import pandas as pd
import sympy
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from dcr.nn import ConceptReasoningLayer
from lens.logic import test_explanation
from lens.models import XReluNN
from lens.utils.base import tree_to_formula, collect_parameters
from lens.utils.relu_nn import get_reduced_model


def counterfactual_dcr(model: ConceptReasoningLayer, c_emb: torch.Tensor, c_score: torch.Tensor, k=5):
    df = {
        "counterfactual_samples": [],
        "counterfactual_preds": [],
        "sample_id": [],
        "iteration": [],
    }

    explanations = model.explain(c_emb, c_score, 'local')
    explanations = pd.DataFrame(explanations)['attention'].tolist()
    for i, (x, c, explanation) in enumerate(zip(c_emb, c_score, explanations)):
        new_c = copy.deepcopy(c)
        orig_class = model(x, new_c.unsqueeze(0))[0].argmax()
        orig_pred = model(x, new_c.unsqueeze(0))[0][orig_class].detach().item()
        if orig_pred < 0.1:
            continue
        df["counterfactual_samples"].append(new_c)
        df["counterfactual_preds"].append(orig_pred/orig_pred)
        df["sample_id"].append(i)
        df["iteration"].append(0)

        if len(explanation) != 0:
            sorted_dimensions = np.argsort(np.abs(explanation))[::-1]
        else:
            sorted_dimensions = np.random.permutation([*range(c_score.shape[1])])
            explanation = torch.rand(c_score.shape[1])
        for d in range(c.shape[0]):
            d_i = sorted_dimensions[d]
            weight = explanation[d_i]
            feat = c[d_i]
            if weight > 0.:
                new_feat = np.max([0, feat - 1])
            else:
                new_feat = np.min([1, feat + 1])
            new_c[d_i] = new_feat
            pred = model(x, new_c.unsqueeze(0))[0][orig_class].detach()
            df["counterfactual_samples"].append(new_c)
            df["counterfactual_preds"].append(pred.item()/orig_pred)
            df["sample_id"].append(i)
            df["iteration"].append(d + 1)
            if d == k:
                break

    return df


def counterfactual_logistic(model: LogisticRegression, x: np.ndarray, k=5):
    df = {
        "counterfactual_samples": [],
        "counterfactual_preds": [],
        "sample_id": [],
        "iteration": [],
    }
    explanations = model.coef_
    if len(explanations) == 1:
        explanations = [- explanations[0], explanations[0]]
    for i, sample in enumerate(x):
        new_sample = copy.deepcopy(sample)
        orig_class = model.predict(new_sample.unsqueeze(0))[0].argmax()
        explanation = explanations[orig_class]
        orig_pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class]
        sorted_dimensions = np.argsort(np.abs(explanation))[::-1]

        df["counterfactual_samples"].append(new_sample)
        df["counterfactual_preds"].append(orig_pred/orig_pred)
        df["sample_id"].append(i)
        df["iteration"].append(0)

        for d in range(sample.shape[0]):
            d_i = sorted_dimensions[d]
            weight = explanation[d_i]
            feat = sample[d_i]
            if weight > 0.:
                new_feat = np.max([0, feat - 1])
            else:
                new_feat = np.min([1, feat + 1])
            new_sample[d_i] = new_feat
            pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class]

            df["counterfactual_samples"].append(new_sample)
            df["counterfactual_preds"].append(pred/orig_pred)
            df["sample_id"].append(i)
            df["iteration"].append(d + 1)
            if d == k:
                break

    return df


def counterfactual_xrelu(model: XReluNN, x: np.ndarray, k=5):
    df = {
        "counterfactual_samples": [],
        "counterfactual_preds": [],
        "sample_id": [],
        "iteration": [],
    }
    for i, sample in enumerate(x):
        new_sample = copy.deepcopy(sample)
        orig_class = model(new_sample.unsqueeze(0))[0].argmax()
        orig_pred = model(new_sample.unsqueeze(0))[0][orig_class].item()

        df["counterfactual_samples"].append(new_sample)
        df["counterfactual_preds"].append(orig_pred/orig_pred)
        df["sample_id"].append(i)
        df["iteration"].append(0)

        reduced_model = get_reduced_model(model.model, sample.squeeze())
        explanations = collect_parameters(reduced_model)[0][0]
        explanation = explanations[orig_class]
        sorted_dimensions = np.argsort(np.abs(explanation))[::-1]

        for d in range(sample.shape[0]):
            d_i = sorted_dimensions[d]
            weight = explanation[d_i]
            feat = sample[d_i]
            if weight > 0.:
                new_feat = np.max([0, feat - 1])
            else:
                new_feat = np.min([1, feat + 1])
            new_sample[d_i] = new_feat
            pred = model(new_sample.unsqueeze(0))[0][orig_class]

            df["counterfactual_samples"].append(new_sample)
            df["counterfactual_preds"].append(pred.item()/orig_pred)
            df["sample_id"].append(i)
            df["iteration"].append(d + 1)

            if d == k:
                break

    return df


def counterfactual_tree(model: DecisionTreeClassifier, x: np.ndarray, y: np.ndarray, k=5):
    n_features = x.shape[1]
    n_classes = y.shape[1]
    concept_names = [f"f_{i}" for i in range(n_features)]
    global_explanations = []
    for target_class in range(n_classes):
        global_explanation = tree_to_formula(model, concept_names, target_class)
        global_explanation = sympy.to_dnf(global_explanation)
        global_explanations.append(global_explanation)

    for sample, label in zip(x, y):
        explanation = ""
        target_class = model.predict(x)
        list_local_explanations = global_explanations[target_class].split("|")
        for local_explanation in list_local_explanations:
            acc = test_explanation(local_explanation, target_class, x, y)
            if acc == 1:
                explanation = local_explanation
                break


counterfactual_functions = {
    "LogisticRegression": counterfactual_logistic,
    "DecisionTreeClassifier": counterfactual_tree,
    "XReluNN": counterfactual_xrelu,
}
