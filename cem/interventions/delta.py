import numpy as np
import torch
from cem.interventions.intervention_policy import InterventionPolicy

class DeltaIntPolicy(InterventionPolicy):

    def __init__(
        self,
        cbm,
        intervened_concepts,
        num_groups_intervened=0,
        **kwargs,
    ):
        self.intervened_concepts = intervened_concepts
        self.num_groups_intervened = num_groups_intervened

    def __call__(
        self,
        x,
        pred_c,
        c,
        y=None,
        competencies=None,
        prev_interventions=None,
        prior_distribution=None,
    ):
        # We have to split it into a list contraction due to the
        # fact that we can't afford to run a np.random.choice
        # that does not allow replacement between samples...
        if prev_interventions is not None:
            mask = prev_interventions.detach().cpu().numpy()
        else:
            mask = np.zeros((x.shape[0], c.shape[-1]), dtype=np.int64)

        if self.num_groups_intervened:
            for concept_idx in self.intervened_concepts:
                mask[:, concept_idx] = 1
        return mask, c
