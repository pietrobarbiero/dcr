import os
import numpy as np
import torch
import pytorch_lightning as pl
from collections import defaultdict
import cem.train.training as cem_train
import logging
import io
from contextlib import redirect_stdout

################################################################################
## HELPER FUNCTIONS
################################################################################

def load_trained_model(
    config,
    n_tasks,
    result_dir,
    n_concepts,
    split=0,
    imbalance=None,
    intervention_idxs=None,
    adversarial_intervention=False,
    train_dl=None,
    sequential=False,
    independent=False,
    gpu=int(torch.cuda.is_available()),
):
    arch_name = config['c_extractor_arch']
    if not isinstance(arch_name, str):
        arch_name = "lambda"
    if split is not None:
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}_"
            f"{arch_name}_fold_{split + 1}"
        )
    else:
        full_run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}_"
            f"{arch_name}"
        )
    selected_concepts = np.arange(n_concepts)
    if (
        (intervention_idxs is not None) and
        (train_dl is not None) and
        (config['architecture'] == "ConceptBottleneckModel") and
        (not config.get('sigmoidal_prob', True))
    ):
        # Then let's look at the empirical distribution of the logits in order
        # to be able to intervene
        model = cem_train.construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
        )
        trainer = pl.Trainer(
            gpus=gpu,
        )
        batch_results = trainer.predict(model, train_dl)
        out_embs = np.concatenate(
            list(map(lambda x: x[1], batch_results)),
            axis=0,
        )
        active_intervention_values = []
        inactive_intervention_values = []
        for idx in range(n_concepts):
            active_intervention_values.append(
                np.percentile(out_embs[:, idx], 95)
            )
            inactive_intervention_values.append(
                np.percentile(out_embs[:, idx], 5)
            )
        if gpu:
            active_intervention_values = torch.cuda.FloatTensor(
                active_intervention_values
            )
            inactive_intervention_values = torch.cuda.FloatTensor(
                inactive_intervention_values
            )
        else:
            active_intervention_values = torch.FloatTensor(
                active_intervention_values
            )
            inactive_intervention_values = torch.FloatTensor(
                inactive_intervention_values
            )
    else:
        active_intervention_values = inactive_intervention_values = None

    if sequential:
        _, c2y_model = cem_train.construct_sequential_models(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
        )
    elif independent:
        _, c2y_model = cem_train.construct_sequential_models(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
        )
    else:
        c2y_model = None
    model = cem_train.construct_model(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        config=config,
        imbalance=imbalance,
        intervention_idxs=intervention_idxs,
        adversarial_intervention=adversarial_intervention,
        active_intervention_values=active_intervention_values,
        inactive_intervention_values=inactive_intervention_values,
        c2y_model=c2y_model,
    )
    if sequential:
        extra = "Sequential"
    elif independent:
        extra = "Independent"
    else:
        extra = ""
    model_saved_path = os.path.join(
        result_dir or ".",
        f'{extra}{full_run_name}.pt'
    )
    model.load_state_dict(torch.load(model_saved_path))
    return model


################################################################################
## CONCEPT INTERVENTION SELECTION POLICIES
################################################################################

def random_int_policy(num_groups_intervened, concept_group_map, config=None):
    selected_groups_for_trial = np.random.choice(
        list(concept_group_map.keys()),
        size=num_groups_intervened,
        replace=False,
    )
    intervention_idxs = []
    for selected_group in selected_groups_for_trial:
        intervention_idxs.extend(concept_group_map[selected_group])
    return sorted(intervention_idxs)


################################################################################
## MAIN INTERVENTION FUNCTION
################################################################################

def intervene_in_cbm(
    config,
    test_dl,
    n_tasks,
    n_concepts,
    result_dir,
    imbalance=None,
    adversarial_intervention=False,
    train_dl=None,
    sequential=False,
    independent=False,
    concept_group_map=None,
    intervened_groups=None,
    gpu=int(torch.cuda.is_available()),
    split=0,
    concept_selection_policy=random_int_policy,
    rerun=False,
    old_results=None,
):
    intervention_accs = []
    # If no concept groups are given, then we assume that all concepts
    # represent a unitary group themselves
    concept_group_map = concept_group_map or dict(
        [(i, [i]) for i in range(n_concepts)]
    )
    groups = intervened_groups or list(range(0, len(concept_group_map) + 1, 1))
    
    if (not rerun) and (old_results is not None):
        for j, num_groups_intervened in enumerate(groups):
            print(
                f"\tTest accuracy when intervening with {num_groups_intervened} "
                f"concept groups is {old_results[j] * 100:.2f}%."
            )
        return old_results
    
    for j, num_groups_intervened in enumerate(groups):
        logging.debug(
            f"Intervening with {num_groups_intervened} out of "
            f"{len(concept_group_map)} concept groups"
        )
        n_trials = config.get('intervention_trials', 5)
        avg = []
        for trial in range(n_trials):
            intervention_idxs = concept_selection_policy(
                num_groups_intervened=num_groups_intervened,
                concept_group_map=concept_group_map,
                config=config,
            )
            logging.debug(
                f"\tFor trial {trial + 1}/{n_trials} for split {split} with "
                f"{num_groups_intervened} groups intervened"
            )
            logging.debug(
                f"\t\tThis results in concepts {intervention_idxs} being "
                f"modified"
            )
            config["shared_prob_gen"] = config.get("shared_prob_gen", False)
            config["per_concept_weight"] = config.get(
                "per_concept_weight",
                False,
            )
            model = load_trained_model(
                config=config,
                n_tasks=n_tasks,
                n_concepts=n_concepts,
                result_dir=result_dir,
                split=split,
                imbalance=imbalance,
                intervention_idxs=intervention_idxs,
                train_dl=train_dl,
                sequential=sequential,
                independent=independent,
            )
            trainer = pl.Trainer(
                gpus=gpu,
            )
            f = io.StringIO()
            with redirect_stdout(f):
                [test_results] = trainer.test(model, test_dl, verbose=False,)
            acc = test_results['test_y_accuracy']
            avg.append(acc)
            logging.debug(
                f"\tFor model at split {split}, intervening with "
                f"{num_groups_intervened} groups (trial {trial + 1}) gives "
                f"test task accuracy {acc * 100:.2f}%."
            )
        print(
            f"\tTest accuracy when intervening with {num_groups_intervened} "
            f"concept groups is "
            f"{np.mean(avg) * 100:.2f}% ± {np.std(avg)* 100:.2f}%."
        )
        intervention_accs.append(np.mean(avg))
    return intervention_accs
