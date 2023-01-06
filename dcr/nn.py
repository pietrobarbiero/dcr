from collections import Counter
import torch

from .semantics import Logic

EPS = 1e-3
MAX_EXP = 30


def softmaxnorm(values, temperature):
    x = torch.clamp_max(values / temperature, MAX_EXP)
    softmax_scores = torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)
    return softmax_scores / softmax_scores.max(dim=1)[0].unsqueeze(1)


class ConceptReasoningLayer(torch.nn.Module):
    def __init__(self, emb_size, n_classes, logic: Logic, temperature_complexity: float = 1.):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.logic = logic
        self.filter_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.sign_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.temperature_complexity = temperature_complexity

    def forward(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)
        # x = c.unsqueeze(-1).repeat(1, 1, self.emb_size)

        if sign_attn is None:
            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence
            sign_attn = torch.sigmoid(self.sign_nn(x))  # TODO: might be independent of input x (but requires OR)

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)   # TODO: Fra check
        sign_terms = self.logic.iff_pair(sign_attn, values)    # TODO: temperature sharp here?

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
           filter_attn = softmaxnorm(self.filter_nn(x), self.temperature_complexity)   # TODO: might be independent of input x (but requires OR)

        # filter values
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(sign_terms, self.logic.neg(filter_attn))

        # generate minterm
        # preds = self.logic.conj(filtered_values, dim=1).squeeze().float()
        preds = self.logic.conj(filtered_values, dim=1).softmax(dim=-1).squeeze()   # FIXME: softmax looks weird

        # TODO: add OR for global explanations

        if return_attn:
            return preds, sign_attn, filter_attn
        else:
            return preds

    def explain(self, x, c, mode):
        assert mode in ['local', 'global', 'exact']
        explanations = None

        y_preds, sign_attn_mask, filter_attn_mask = self.forward(x, c, return_attn=True)
        sign_attn_mask, filter_attn_mask = sign_attn_mask > 0.5, filter_attn_mask > 0.5

        # TODO: add extraction of exact fuzzy rules (fidelity=1)

        if mode in ['local', 'global']:
            # extract local explanations
            predictions = y_preds.argmax(dim=-1).detach()
            explanations = []
            all_class_explanations = {c: [] for c in range(self.n_classes)}
            for filter_attn, sign_attn, prediction in zip(filter_attn_mask, sign_attn_mask, predictions):
                # select mask for predicted class only
                # and generate minterm
                minterm = []
                for idx, (concept_score, attn_score) in enumerate(zip(sign_attn[:, prediction], filter_attn[:, prediction])):
                    if attn_score:
                        if concept_score:
                            minterm.append(f'f{idx}')
                        else:
                            minterm.append(f'~f{idx}')
                minterm = ' & '.join(minterm)
                # add explanation to list
                all_class_explanations[prediction.item()].append(minterm)
                explanations.append({
                    'class': prediction.item(),
                    'explanation': minterm,
                })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations
