import copy
import torch
from torch.nn.functional import one_hot


def counterfact(model, x, c):
    # find original predictions
    if isinstance(model, torch.nn.Module):
        old_preds = model.forward(x, c)
    else:
        old_preds = model.predict(c)
        old_preds = one_hot(torch.LongTensor(old_preds))

    # find a (random) counterfactual: a (random) perturbation of the input that would change the prediction
    counterfactuals = {'sample_id': [], 'old_pred': [], 'new_pred': [], 'old_concepts': [], 'new_concepts': []}
    for sid, (old_concept_emb, old_concept_score, old_pred) in enumerate(zip(x, c, old_preds)):
        old_concept_emb, old_concept_score = old_concept_emb.unsqueeze(0), old_concept_score.unsqueeze(0)
        new_concept_score = copy.deepcopy(old_concept_score)
        new_concept_emb = copy.deepcopy(old_concept_emb)
        target_pred = (1 - old_pred).argmax(dim=-1)

        # select a random sequence of concepts to perturb
        rnd_concept_idxs = torch.randperm(c.shape[1])
        for rnd_concept_idx in rnd_concept_idxs:
            # perturb concept score
            new_concept_score[:, rnd_concept_idx] = 1 - old_concept_score[:, rnd_concept_idx]
            # FIXME: if we have access to CEM model, we change the embedding by just flipping the concept score!
            # perturb concept embedding
            new_concept_emb[:, rnd_concept_idx] = torch.mean(x[old_preds.argmax(dim=-1) == target_pred, rnd_concept_idx], dim=0)

            # TODO: rule intervention? update attention weights according to new concept scores
            # new_sign_attn = (1 - new_concept_score).unsqueeze(-1).repeat(1, 1, self.n_classes)
            # new_sign_attn[:, :, target_pred] = new_concept_score
            # generate new prediction

            # find new predictions
            if isinstance(model, torch.nn.Module):
                new_pred = model.forward(new_concept_emb, new_concept_score)  # TODO: we may start with original attn and then make all concepts available
            else:
                new_pred = model.predict(new_concept_score)
                new_pred = one_hot(torch.LongTensor(new_pred))

            # if new predictions match the target class, then we found the counterfactual!
            if new_pred.argmax(dim=-1) == target_pred:
                counterfactuals['sample_id'].append(sid)
                counterfactuals['old_pred'].append(old_pred.tolist())
                counterfactuals['new_pred'].append(new_pred.tolist())
                counterfactuals['old_concepts'].append(old_concept_score.tolist()[0])
                counterfactuals['new_concepts'].append(new_concept_score.tolist()[0])
                break

    return counterfactuals
