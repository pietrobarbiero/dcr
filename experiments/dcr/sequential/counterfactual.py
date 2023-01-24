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
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
from xgboost import XGBClassifier

from dcr.nn import ConceptReasoningLayer
from lens.logic import test_explanation
from lens.models import XReluNN
from lens.utils.base import tree_to_formula, collect_parameters
from lens.utils.relu_nn import get_reduced_model


def counterfactual_dcr(model: ConceptReasoningLayer, c_emb: torch.Tensor, c_score: torch.Tensor, k=5):
    df = {
        "counterfactual_samples": [],
        "counterfactual_preds": [],
        "counterfactual_preds_norm": [],
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
        df["counterfactual_samples"].append(copy.deepcopy(new_c))
        df["counterfactual_preds"].append(orig_pred)
        df["counterfactual_preds_norm"].append(orig_pred/orig_pred)
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
            df["counterfactual_samples"].append(copy.deepcopy(new_c))
            df["counterfactual_preds"].append(pred.item())
            df["counterfactual_preds_norm"].append(pred.item()/orig_pred)
            df["sample_id"].append(i)
            df["iteration"].append(d + 1)
            if d == k:
                break

    return df


def counterfactual_xrelu(model: XReluNN, x: np.ndarray, k=5):
    df = {
        "counterfactual_samples": [],
        "counterfactual_preds": [],
        "counterfactual_preds_norm": [],
        "sample_id": [],
        "iteration": [],
    }
    for i, sample in enumerate(x):
        new_sample = copy.deepcopy(sample)
        orig_class = model(new_sample.unsqueeze(0))[0].argmax()
        orig_pred = model(new_sample.unsqueeze(0))[0][orig_class].item()

        df["counterfactual_samples"].append(copy.deepcopy(new_sample))
        df["counterfactual_preds"].append(orig_pred)
        df["counterfactual_preds_norm"].append(orig_pred/orig_pred)
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

            df["counterfactual_samples"].append(copy.deepcopy(new_sample))
            df["counterfactual_preds"].append(pred.item())
            df["counterfactual_preds_norm"].append(pred.item()/orig_pred)
            df["sample_id"].append(i)
            df["iteration"].append(d + 1)

            if d == k:
                break

    return df


def counterfactual_logistic(model: LogisticRegression, x: np.ndarray, k=5):
    df = {
        "counterfactual_samples": [],
        "counterfactual_preds": [],
        "counterfactual_preds_norm": [],
        "sample_id": [],
        "iteration": [],
    }
    explanations = model.coef_
    if len(explanations) == 1:
        explanations = [- explanations[0], explanations[0]]
    for i, sample in enumerate(x):
        new_sample = copy.deepcopy(sample)
        orig_class = model.predict(new_sample.unsqueeze(0))[0]
        explanation = explanations[orig_class]
        orig_pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class]
        sorted_dimensions = np.argsort(np.abs(explanation))[::-1]

        df["counterfactual_samples"].append(copy.deepcopy(new_sample))
        df["counterfactual_preds"].append(orig_pred)
        df["counterfactual_preds_norm"].append(orig_pred/orig_pred)
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
            assert pred <= orig_pred, "Error in finding counterfactual example for logistic regression classifier"

            df["counterfactual_samples"].append(copy.deepcopy(new_sample))
            df["counterfactual_preds"].append(pred)
            df["counterfactual_preds_norm"].append(pred/ orig_pred)
            df["sample_id"].append(i)
            df["iteration"].append(d + 1)
            if d == k:
                break

    return df


def counterfactual_tree(model: DecisionTreeClassifier, x: np.ndarray, k=5):
    tree_ = model.tree_

    df = {
        "counterfactual_samples": [],
        "counterfactual_preds": [],
        "counterfactual_preds_norm": [],
        "sample_id": [],
        "iteration": [],
    }
    for i, sample in enumerate(x):
        global new_sample
        new_sample = copy.deepcopy(sample)
        orig_class = model.predict(new_sample.unsqueeze(0))[0]
        orig_pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class]
        df["counterfactual_samples"].append(copy.deepcopy(new_sample))
        df["counterfactual_preds"].append(orig_pred)
        df["counterfactual_preds_norm"].append(orig_pred/orig_pred)
        df["sample_id"].append(i)
        df["iteration"].append(0)

        node = 0
        while tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_value = sample[tree_.feature[node]]
            if feature_value < tree_.threshold[node]:
                node = tree_.children_left[node]
            else:
                node = tree_.children_right[node]

        global searched
        global pred
        pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class]

        for d in range(sample.shape[0]):
            searched = np.zeros(tree_.node_count)
            def find_counterfactual_tree(node_idx, cls):
                # check if node already analyzed
                if searched[node_idx] == 1:
                    return False
                searched[node_idx] = 1

                # check if node is a leaf
                if tree_.feature[node_idx] == _tree.TREE_UNDEFINED:
                    node_cls = tree_.value[node_idx].squeeze().argmax()
                    prob = tree_.value[node_idx].squeeze()[cls] /tree_.value[node_idx].squeeze().sum()
                    if (node_cls != cls).item() & (prob < pred).item():
                        return True

                # analyze left branch
                if tree_.children_left[node_idx] != _tree.TREE_LEAF:
                    if find_counterfactual_tree(tree_.children_left[node_idx], cls):
                        feature = tree_.feature[node_idx].item()
                        new_sample[feature] = 0.0
                        return True

                # analyze right branch
                if tree_.children_right[node_idx] != _tree.TREE_LEAF:
                    if find_counterfactual_tree(tree_.children_right[node_idx], cls):
                        feature = tree_.feature[node_idx]
                        new_sample[feature] = 1.0
                        return True

                # analyze parent
                parent_node = 0
                if len(np.argwhere(tree_.children_left == node_idx)) == 1:
                    parent_node = np.argwhere((tree_.children_left == node_idx).squeeze()).item()
                elif len(np.argwhere(tree_.children_right == node_idx)) == 1:
                    parent_node = np.argwhere((tree_.children_right == node_idx).squeeze()).item()
                if find_counterfactual_tree(parent_node, cls):
                    return True

                return False

            assert orig_class == tree_.value[node].squeeze().argmax()
            found = find_counterfactual_tree(node, orig_class)

            class_pred = model.predict(new_sample.unsqueeze(0))[0]
            assert class_pred != orig_class, "Error in finding counterfactual example for decision tree classifier"
            assert model.predict_proba(new_sample.unsqueeze(0))[0][orig_class] <= pred, \
                "Error in finding counterfactual example for decision tree classifier"
            pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class]

            df["counterfactual_samples"].append(copy.deepcopy(new_sample))
            df["counterfactual_preds"].append(pred)
            df["counterfactual_preds_norm"].append(pred/orig_pred)
            df["sample_id"].append(i)
            df["iteration"].append(d + 1)


    return df


# from matplotlib import pyplot as plt
# from sklearn import tree
#
# fig = plt.figure(figsize=(25, 20))
# _ = tree.plot_tree(model,
#                    class_names=["0", "1"])
# plt.show()


def counterfactual_xgboost(model: XGBClassifier, x: np.ndarray, k=5):
    df = {
        "counterfactual_samples": [],
        "counterfactual_preds": [],
        "counterfactual_preds_norm": [],
        "sample_id": [],
        "iteration": [],
    }
    for i, sample in enumerate(x):
        new_sample = copy.deepcopy(sample)
        orig_class = model.predict(new_sample.unsqueeze(0))[0]
        orig_pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class]
        sorted_dimensions = np.random.permutation([*range(sample.shape[0])])

        df["counterfactual_samples"].append(copy.deepcopy(new_sample))
        df["counterfactual_preds"].append(orig_pred)
        df["counterfactual_preds_norm"].append(orig_pred/orig_pred)
        df["sample_id"].append(i)
        df["iteration"].append(0)

        for d in range(sample.shape[0]):
            d_i = sorted_dimensions[d]
            weight = np.random.rand()
            feat = sample[d_i]
            if weight > 0.:
                new_feat = np.max([0, feat - 1])
            else:
                new_feat = np.min([1, feat + 1])
            new_sample[d_i] = new_feat
            pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class]

            df["counterfactual_samples"].append(copy.deepcopy(new_sample))
            df["counterfactual_preds"].append(pred)
            df["counterfactual_preds_norm"].append(pred/ orig_pred)
            df["sample_id"].append(i)
            df["iteration"].append(d + 1)
            if d == k:
                break

    return df


def counterfactual_lime(model: XGBClassifier, x: torch.Tensor, k=5):
    df = {
        "counterfactual_samples": [],
        "counterfactual_preds": [],
        "counterfactual_preds_norm": [],
        "sample_id": [],
        "iteration": [],
    }

    num_features = x.shape[1]
    explainer = LimeTabularExplainer(x.numpy(), categorical_features=[*range(num_features)],
                                     feature_names=[f"x_{i}" for i in range(num_features)])
    for i, sample in enumerate(x):
        new_sample = copy.deepcopy(sample)
        orig_class = model.predict(new_sample.unsqueeze(0))[0]
        orig_pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class].item()
        df["counterfactual_samples"].append(copy.deepcopy(new_sample))
        df["counterfactual_preds"].append(orig_pred)
        df["counterfactual_preds_norm"].append(orig_pred / orig_pred)
        df["sample_id"].append(i)
        df["iteration"].append(0)
        lime_explanation = explainer.explain_instance(sample.cpu().detach().numpy().squeeze(), model.predict_proba,
                                                 num_features=num_features, top_labels=1).as_list(label=orig_class)
        explanation = np.zeros(num_features)
        for feat, imp in lime_explanation:
            d = int(feat[2])
            sign = int(feat[4])
            if not sign:
                imp = - imp
            explanation[d] = imp

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
            pred = model.predict_proba(new_sample.unsqueeze(0))[0][orig_class]

            df["counterfactual_samples"].append(copy.deepcopy(new_sample))
            df["counterfactual_preds"].append(pred.item())
            df["counterfactual_preds_norm"].append(pred.item()/orig_pred)
            df["sample_id"].append(i)
            df["iteration"].append(d + 1)

            if d == k:
                break

    return df


counterfactual_functions = {
    "XReluNN": counterfactual_xrelu,
    "LogisticRegression": counterfactual_logistic,
    "DecisionTreeClassifier": counterfactual_tree,
    "XGBClassifier": counterfactual_xgboost,
}

