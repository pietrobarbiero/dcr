import copy
import numpy as np
import os
import joblib
import pytorch_lightning as pl
import torch

from torchvision.models import resnet18, resnet34, resnet50, densenet121

import cem.models.adversarial_cbm as adversarial_cbm
import cem.models.backtracking as backtrack
import cem.models.cbm as models_cbm
import cem.models.cem as models_cem
import cem.models.defer_cem as defer_cem
import cem.models.direction_cem as direction_cem
import cem.models.global_bank_cem as models_global_mixcem
import cem.models.hybrid_cem as models_hcem
import cem.models.intcbm as models_intcbm
import cem.models.mixcem as models_mixcem
import cem.models.posthoc_cbm as models_pcbm
import cem.models.probcbm as models_probcbm
import cem.models.standard as standard_models
import cem.models.global_approx_cem as models_global_approx
import cem.models.separator_cem as separator_cem
import cem.models.prob_cem as prob_cem
import cem.models.certificate_cem as certificate_cem
import cem.train.utils as utils


################################################################################
## HELPER LAYERS
################################################################################


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

################################################################################
## MODEL CONSTRUCTION
################################################################################


def construct_model(
    n_concepts,
    n_tasks,
    config,
    c2y_model=None,
    x2c_model=None,
    imbalance=None,
    task_class_weights=None,
    intervention_policy=None,
    active_intervention_values=None,
    inactive_intervention_values=None,
    output_latent=False,
    output_interventions=False,
    train=False,
):
    task_loss_weight = config.get('task_loss_weight', 1.0)
    if config["architecture"] in ["ConceptEmbeddingModel", "CEM"]:
        model_cls = models_cem.ConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "shared_prob_gen": config.get("shared_prob_gen", True),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu"
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
        }
        if "embeding_activation" in config:
            # Legacy support for typo in argument
            extra_params["embedding_activation"] = config["embeding_activation"]

    elif (
        config["architecture"] in ["HybridConceptEmbeddingModel", "H-CEM"]
    ):
        model_cls = models_hcem.HybridConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "constant_emb_proportion": config.get(
                "constant_emb_proportion",
                0.5,
            ),
            "shared_prob_gen": config.get("shared_prob_gen", True),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "const_embedding_activation": config.get(
                "const_embedding_activation",
                "leakyrelu"
            ),
            "dyn_embedding_activation": config.get(
                "dyn_embedding_activation",
                "leakyrelu"
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
            "contrastive_anchors": config.get(
                "contrastive_anchors",
                False,
            ),
        }

    elif (
        config["architecture"] in ["MultiConceptEmbeddingModel", "Multi-CEM"]
    ):
        model_cls = models_hcem.MultiConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "n_discovered_embs": config.get(
                "n_discovered_embs",
                4,
            ),
            "contrastive_loss_weight": config.get(
                "contrastive_loss_weight",
                0.0,
            ),
            "mix_ground_truth_embs": config.get(
                "mix_ground_truth_embs",
                True,
            ),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu"
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
        }
    elif config['architecture'] in [
        'GlobalBankConceptEmbeddingModel',
        'BankMixCEM',
        'GlobalBankCEM',
        'GlobalBankMixCEM',
    ]:
        model_cls = models_global_mixcem.GlobalBankConceptEmbeddingModel
        extra_params = {
            'rollout_init_steps': config.get('rollout_init_steps', 0),
            'int_model_layers': config.get('int_model_layers', None),
            'int_model_use_bn': config.get('int_model_use_bn', True),
            'num_rollouts': config.get('num_rollouts', 1),
            'intervention_discount': config.get('intervention_discount', 1),
            'include_only_last_trajectory_loss': config.get('include_only_last_trajectory_loss', True),
            'intervention_task_loss_weight': config.get('intervention_task_loss_weight', 1),
            'intervention_weight': config.get('intervention_weight', 5),
            'concept_map': config.get('concept_map', None),
            'max_horizon': config.get('max_horizon', 6),
            'initial_horizon': config.get('initial_horizon', 2),
            'horizon_rate': config.get('horizon_rate', 1.005),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                1,
            ),

            "emb_size": config["emb_size"],
            "shared_emb_generator": config.get(
                "shared_emb_generator",
                True,
            ),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                None,
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),



            "n_concept_variants": config.get(
                "n_concept_variants",
                5,
            ),
            "temperature": config.get(
                "temperature",
                1,
            ),
            "selection_mode": config.get(
                'selection_mode',
                'attention',
            ),
            "soft_select": config.get(
                'soft_select',
                True,
            ),
            "learnable_prob": config.get(
                'learnable_prob',
                False,
            ),
            "remap_context": config.get(
                'remap_context',
                False,
            ),
            "add_dynamic_embedding": config.get(
                'add_dynamic_embedding',
                False,
            ),
            "dynamic_emb_concept_loss_weight": config.get(
                'dynamic_emb_concept_loss_weight',
                0
            ),
            "distance_selection": config.get(
                'distance_selection',
                'hard',
            ),
            "fixed_embeddings": config.get('fixed_embeddings', False),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "fixed_scale": config.get('fixed_scale', None),
            "bottleneck_pooling": config.get('bottleneck_pooling', 'concat'),
        }

    elif config["architecture"] in [
        "ResidualMixingConceptEmbeddingModel",
        "ResidualMixCEM",
        "MixingConceptEmbeddingModel",
        "MixCEM",
    ]:
        if "Residual" in config["architecture"]:
            model_cls = models_hcem.ResidualMixingConceptEmbeddingModel
        else:
            model_cls = models_hcem.MixingConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "n_discovered_concepts": config.get(
                "n_discovered_concepts",
                4,
            ),
            "contrastive_loss_weight": config.get(
                "contrastive_loss_weight",
                0.0,
            ),
            "shared_emb_generator": config.get(
                "shared_emb_generator",
                False,
            ),
            "normalize_embs": config.get(
                "normalize_embs",
                False,
            ),
            "sample_probs": config.get(
                "sample_probs",
                False,
            ),
            "cond_discovery": config.get(
                "cond_discovery",
                False,
            ),
            "intermediate_task_concept_loss": config.get(
                "intermediate_task_concept_loss",
                0.0,
            ),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                1,
            ),
            "mix_ground_truth_embs": config.get(
                "mix_ground_truth_embs",
                True,
            ),
            "discovered_probs_entropy": config.get(
                "discovered_probs_entropy",
                0,
            ),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "dyn_training_intervention_prob": config.get(
                "dyn_training_intervention_prob",
                0,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu"
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "fixed_embeddings": config.get('fixed_embeddings', False),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "use_cosine_similarity": config.get('use_cosine_similarity', False),
            "use_linear_emb_layer": config.get('use_linear_emb_layer', False),
            "fixed_scale": config.get('fixed_scale', None),
        }

    elif config["architecture"] in [
        "ProbCBM",
        "ProbabilisticConceptBottleneckModel",
        "ProbabilisticCBM",
    ]:
        model_cls = models_probcbm.ProbCBM
        extra_params = dict(
            lr_ratio=config.get(
                'lr_ratio',
                10,
            ),
            hidden_dim=config.get(
                'hidden_dim',
                16,
            ),
            class_hidden_dim=config.get(
                'class_hidden_dim',
                128,
            ),
            intervention_prob=config.get(
                'intervention_prob',
                0.5,
            ),
            use_class_emb_from_concept=config.get(
                'use_class_emb_from_concept',
                False,
            ),
            use_probabilistic_concept=config.get(
                'use_probabilistic_concept',
                True,
            ),
            pretrained=config.get(
                'pretrained',
                True,
            ),
            n_samples_inference=config.get(
                'n_samples_inference',
                50,
            ),
            use_neg_concept=config.get(
                'use_neg_concept',
                True,
            ),
            pred_class=config.get(
                'pred_class',
                True,
            ),
            use_scale=config.get(
                'use_scale',
                True,
            ),
            activation_concept2class=config.get(
                'activation_concept2class',
                'prob',
            ),
            token2concept=config.get(
                'token2concept',
                None,
            ),
            train_class_mode=config.get(
                'train_class_mode',
                'sequential',
            ),
            init_negative_scale=config.get(
                'init_negative_scale',
                5,
            ),
            init_shift=config.get(
                'init_shift',
                5,
            ),
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            use_concept_groups=config.get(
                'use_concept_groups',
                False,
            ),
            vib_beta=config.get(
                'vib_beta',
                0.00005,
            )
        )
    elif config["architecture"] in ["IntAwareConceptBottleneckModel", "IntCBM"]:
        task_loss_weight = config.get('task_loss_weight', 0.0)
        model_cls = models_intcbm.IntAwareConceptBottleneckModel
        extra_params = {
            "bool": config["bool"],
            "extra_dims": config["extra_dims"],
            "sigmoidal_extra_capacity": config.get(
                "sigmoidal_extra_capacity",
                True,
            ),
            "sigmoidal_prob": config.get("sigmoidal_prob", True),
            "intervention_policy": intervention_policy,
            "bottleneck_nonlinear": config.get("bottleneck_nonlinear", None),
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "x2c_model": x2c_model,
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "concept_map": config.get("concept_map", None),
            "max_horizon": config.get("max_horizon", 6),
            "include_only_last_trajectory_loss": config.get(
                "include_only_last_trajectory_loss",
                True,
            ),
            "intervention_task_loss_weight": config.get(
                "intervention_task_loss_weight",
                1,
            ),
            "initial_horizon": config.get("initial_horizon", 2),
            "use_concept_groups": config.get("use_concept_groups", True),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                config.get("intervention_task_discount", 1.1),
            ),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", True),
            "num_rollouts": config.get("num_rollouts", 1),
        }
    elif config["architecture"] in ["IntAwareConceptEmbeddingModel", "IntCEM"]:
        task_loss_weight = config.get('task_loss_weight', 1)
        model_cls = models_intcbm.IntAwareConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu",
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "concept_map": config.get("concept_map", None),
            "max_horizon": config.get("max_horizon", 6),
            "include_only_last_trajectory_loss": config.get(
                "include_only_last_trajectory_loss",
                True,
            ),
            "intervention_task_loss_weight": config.get(
                "intervention_task_loss_weight",
                1,
            ),
            "initial_horizon": config.get("initial_horizon", 2),
            "use_concept_groups": config.get("use_concept_groups", False),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                config.get("intervention_task_discount", 1.1),
            ),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", False),
            "num_rollouts": config.get("num_rollouts", 1),
        }

    elif config['architecture'] in [
        'SeparatorConceptEmbeddingModel',
        'SeparatorCEM',
    ]:
        task_loss_weight = config.get('task_loss_weight', 1)
        model_cls = separator_cem.SeparatorConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu",
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "concept_map": config.get("concept_map", None),
            "max_horizon": config.get("max_horizon", 6),
            "include_only_last_trajectory_loss": config.get(
                "include_only_last_trajectory_loss",
                True,
            ),
            "intervention_task_loss_weight": config.get(
                "intervention_task_loss_weight",
                1,
            ),
            "initial_horizon": config.get("initial_horizon", 2),
            "use_concept_groups": config.get("use_concept_groups", False),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                config.get("intervention_task_discount", 1.1),
            ),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", False),
            "num_rollouts": config.get("num_rollouts", 1),


            # New parameters
            "temperature": config.get('temperature', 1),
            "n_concept_variants": config.get('n_concept_variants', 5),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "fixed_embeddings": config.get('fixed_embeddings', False),
            "attention_fn": config.get('attention_fn', 'softmax'),
            "margin_loss_weight": config.get(
                'margin_loss_weight',
                0,
            ),
            "ood_dropout_prob": config.get(
                "ood_dropout_prob",
                0,
            ),
            "separator_warmup_steps": config.get(
                'separator_warmup_steps',
                0,
            ),
            "box_temperature": config.get(
                'box_temperature',
                10,
            ),
            "bounds_loss_weight": config.get(
                'bounds_loss_weight',
                10,
            ),
            "init_bound_val": config.get(
                'init_bound_val',
                5,
            ),
            "pooling_mode": config.get(
                'pooling_mode',
                'concat',
            ),
            "sep_loss_weight": config.get(
                'sep_loss_weight',
                0,
            ),
            "selection_mode": config.get(
                'selection_mode',
                'prob_box',
            ),
            "projection_dim": config.get(
                'projection_dim',
                None,
            ),
            "separator_mode": config.get(
                'separator_mode',
                "individual",
            ),
        }



    elif config['architecture'] in [
        'ProbConceptEmbeddingModel',
        'ProbCEM',
    ]:
        task_loss_weight = config.get('task_loss_weight', 1)
        model_cls = prob_cem.ProbConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu",
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "concept_map": config.get("concept_map", None),
            "max_horizon": config.get("max_horizon", 6),
            "include_only_last_trajectory_loss": config.get(
                "include_only_last_trajectory_loss",
                True,
            ),
            "intervention_task_loss_weight": config.get(
                "intervention_task_loss_weight",
                1,
            ),
            "initial_horizon": config.get("initial_horizon", 2),
            "use_concept_groups": config.get("use_concept_groups", False),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                config.get("intervention_task_discount", 1.1),
            ),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", False),
            "num_rollouts": config.get("num_rollouts", 1),


            # New parameters
            "temperature": config.get(
                'temperature',
                1,
            ),
            "n_concept_variants": config.get(
                'n_concept_variants',
                5,
            ),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "fixed_embeddings": config.get(
                'fixed_embeddings',
                False,
            ),
            "initial_log_variances": config.get(
                'initial_log_variances',
                None,
            ),
            "fixed_variances": config.get(
                'fixed_variances',
                False,
            ),
            "attention_fn": config.get(
                'attention_fn',
                'softmax',
            ),
            "ood_dropout_prob": config.get(
                'ood_dropout_prob',
                0,
            ),
            "pooling_mode": config.get(
                'pooling_mode',
                'concat',
            ),
            "selection_mode": config.get(
                'selection_mode',
                'z_score',
            ),
            "kl_loss_weight": config.get(
                'kl_loss_weight',
                0,
            ),
            "box_temparature": config.get(
                'box_temparature',
                1,
            ),
            "threshold": config.get(
                'threshold',
                None,
            ),
            "learnable_concept_embs": config.get(
                'learnable_concept_embs',
                True,
            ),
            "shared_attn_module": config.get(
                'shared_attn_module',
                False,
            ),
            "global_above_thresh": config.get(
                'global_above_thresh',
                True,
            ),
        }



    elif config['architecture'] in [
        'CertificateConceptEmbeddingModel',
        'CertificateCEM',
    ]:
        task_loss_weight = config.get('task_loss_weight', 1)
        model_cls = certificate_cem.CertificateConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu",
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "concept_map": config.get("concept_map", None),
            "max_horizon": config.get("max_horizon", 6),
            "include_only_last_trajectory_loss": config.get(
                "include_only_last_trajectory_loss",
                True,
            ),
            "intervention_task_loss_weight": config.get(
                "intervention_task_loss_weight",
                1,
            ),
            "initial_horizon": config.get("initial_horizon", 2),
            "use_concept_groups": config.get("use_concept_groups", False),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                config.get("intervention_task_discount", 1.1),
            ),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", False),
            "num_rollouts": config.get("num_rollouts", 1),

            # New parameters
            "temperature": config.get('temperature', 1),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "fixed_embeddings": config.get('fixed_embeddings', False),
            "ood_dropout_prob": config.get('ood_dropout_prob', 0),
            "pooling_mode": config.get('pooling_mode', 'concat'),
            "certificate_loss_weight": config.get('certificate_loss_weight', 0),
            "selection_mode": config.get("selection_mode", "individual"),
            "hard_eval_selection": config.get("hard_eval_selection", None),
            "selection_sample": config.get("selection_sample", False),
            "eval_majority_vote": config.get('eval_majority_vote', False),
            "mixed_probs": config.get('mixed_probs', False),
            "contrastive_reg": config.get('contrastive_reg', 0),
            "global_ood_prob": config.get('global_ood_prob', 0),
            "init_dyn_temps": config.get('init_dyn_temps', 1.5),
            "init_global_temps": config.get('init_global_temps', 0.5),
            "global_temp_reg": config.get('global_temp_reg', 0),
            "max_temperature": config.get("max_temperature", 1),
            "inference_dyn_prob": config.get('inference_dyn_prob', False),
            "learnable_temps": config.get("learnable_temps", False),
            "positive_calibration": config.get("positive_calibration", False),
            "class_wise_temperature": config.get('class_wise_temperature', True),
            "entire_global_prob": config.get('entire_global_prob', 0),
            "counter_limit": config.get('counter_limit', 0),
            "print_eval_only": config.get('print_eval_only', True),
            "hard_selection_value": config.get('hard_selection_value', None),
            "threshold_probs": config.get('threshold_probs', None),
            "global_prediction_reg": config.get('global_prediction_reg', 0),
            "hard_train_selection": config.get('hard_train_selection', None),
            "train_prob_thresh": config.get('train_prob_thresh', None),
            "random_selection_prob": config.get('random_selection_prob', 0),
        }

    elif config['architecture'] in [
        'GlobalApproxConceptEmbeddingModel',
        'GlobalApproxCEM',
    ]:
        task_loss_weight = config.get('task_loss_weight', 1)
        model_cls = models_global_approx.GlobalApproxConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu",
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "concept_map": config.get("concept_map", None),
            "max_horizon": config.get("max_horizon", 6),
            "include_only_last_trajectory_loss": config.get(
                "include_only_last_trajectory_loss",
                True,
            ),
            "intervention_task_loss_weight": config.get(
                "intervention_task_loss_weight",
                1,
            ),
            "initial_horizon": config.get("initial_horizon", 2),
            "use_concept_groups": config.get("use_concept_groups", False),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                config.get("intervention_task_discount", 1.1),
            ),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", False),
            "num_rollouts": config.get("num_rollouts", 1),


            # New parameters
            "temperature": config.get('temperature', 1),
            "n_concept_variants": config.get('n_concept_variants', 5),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "fixed_embeddings": config.get('fixed_embeddings', False),
            "mode": config.get('mode', 'ood_same'),
            "l2_dist_loss_weight": config.get('l2_dist_loss_weight', 0),
            "compression_mode": config.get('compression_mode', 'learnt'),
            "attention_fn": config.get('attention_fn', 'sigmoid'),
            "use_dynamic_for_probs": config.get(
                'use_dynamic_for_probs',
                False,
            ),
            "log_thresholds": config.get(
                "log_thresholds",
                True,
            ),
            "distance_l2_loss": config.get(
                'distance_l2_loss',
                0,
            ),

            # OOD stuff
            "thresh_l2_loss": config.get(
                'thresh_l2_loss',
                0,
            ),
            "ood_dropout_prob": config.get(
                "ood_dropout_prob",
                0,
            ),
            "approx_prediction_mode": config.get(
                'approx_prediction_mode',
                'same',
            ),
            "global_mixture_components": config.get(
                'global_mixture_components',
                None,
            ),
            "ood_loss_weight": config.get(
                'ood_loss_weight',
                0,
            ),
        }

        run_name = config["run_name"]
        split = config.get("split", 0)
        if split is not None:
            full_run_name = (
                f"{run_name}_fold_{split + 1}"
            )
        else:
            full_run_name = (
                f"{run_name}"
            )
        result_dir = config.get("result_dir", ".")

        gmm_model_path = os.path.join(
            result_dir or ".",
            f'{full_run_name}_GMM_model.pt'
        )

        if os.path.exists(gmm_model_path):
            extra_params['gmms'] = joblib.load(gmm_model_path)

        gmm_threshold_path = os.path.join(
            result_dir or ".",
            f'{full_run_name}_GMM_threshold.pt'
        )
        if os.path.exists(gmm_threshold_path):
            extra_params['gmm_thresholds'] = joblib.load(gmm_threshold_path)

        prob_gmm_model_path = os.path.join(
            result_dir or ".",
            f'{full_run_name}_Prob_GMM_model.pt'
        )
        if os.path.exists(prob_gmm_model_path):
            extra_params['prob_gmms'] = joblib.load(prob_gmm_model_path)

        prob_gmm_threshold_path = os.path.join(
            result_dir or ".",
            f'{full_run_name}_Prob_GMM_threshold.pt'
        )
        if os.path.exists(prob_gmm_threshold_path):
            extra_params['prob_threshs'] = joblib.load(prob_gmm_threshold_path)


    elif config["architecture"] in ["IntAwareMixCEM"]:
        task_loss_weight = config.get('task_loss_weight', 1)
        model_cls = backtrack.IntAwareMixCEM
        extra_params = {
            "emb_size": config["emb_size"],
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                "leakyrelu",
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "intervention_weight": config.get("intervention_weight", 5),
            "horizon_rate": config.get("horizon_rate", 1.005),
            "concept_map": config.get("concept_map", None),
            "max_horizon": config.get("max_horizon", 6),
            "include_only_last_trajectory_loss": config.get(
                "include_only_last_trajectory_loss",
                True,
            ),
            "intervention_task_loss_weight": config.get(
                "intervention_task_loss_weight",
                1,
            ),
            "initial_horizon": config.get("initial_horizon", 2),
            "use_concept_groups": config.get("use_concept_groups", False),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                config.get("intervention_task_discount", 1.1),
            ),
            "rollout_init_steps": config.get('rollout_init_steps', 0),
            "int_model_layers": config.get("int_model_layers", None),
            "int_model_use_bn": config.get("int_model_use_bn", False),
            "num_rollouts": config.get("num_rollouts", 1),



            "fixed_embeddings": config.get('fixed_embeddings', False),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "residual_scale": config.get('residual_scale', None),
            "learnable_residual_scale": config.get('learnable_residual_scale', True),
            "residual_scale_reg": config.get('residual_scale_reg', 0),
            "sigmoidal_residual_scale": config.get('sigmoidal_residual_scale', False),
            "residual_scale_norm_metric": config.get('residual_scale_norm_metric', 1),
            "normalize_residual": config.get('normalize_residual', False),
            "residual_nll_reg": config.get('residual_nll_reg', 0),
            "fixed_residual_scale": config.get('fixed_residual_scale', None),
            "intermediate_task_concept_loss": config.get('intermediate_task_concept_loss', 0),
            "drop_residual_prob": config.get('drop_residual_prob', 0),
            "scalar_residual": config.get('scalar_residual', False),
            "sigmoidal_residual": config.get('sigmoidal_residual', False),
            "use_distance_probs": config.get('use_distance_probs', False),
        }

    elif (
        "AdversarialConceptBottleneckModel" in config["architecture"] or
        "AdversarialCBM" in config["architecture"]
    ):
        model_cls = adversarial_cbm.AdversarialConceptBottleneckModel
        extra_params = {
            "bool": config["bool"],
            "extra_dims": config["extra_dims"],
            "sigmoidal_extra_capacity": config.get(
                "sigmoidal_extra_capacity",
                True,
            ),
            "sigmoidal_prob": config.get("sigmoidal_prob", True),
            "intervention_policy": intervention_policy,
            "bottleneck_nonlinear": config.get("bottleneck_nonlinear", None),
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "x2c_model": x2c_model,
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
            "discriminator_layers": config.get('discriminator_layers', []),
            "discriminator_loss_weight": config.get('discriminator_loss_weight', 1),
            "interleave_steps": config.get('interleave_steps', 5),
        }

    elif config["architecture"] in [
        "PCBM",
        "PosthocCBM",
        "Post-hocCBM",
        "PosthocConceptBottleneckModel",
        "Post-hocConceptBottleneckModel",
    ]:
        if config.get('pretrained_model', None) is None:
            blackbox_model_config = config["blackbox_model_config"]
            bbox, _ = standard_models.construct_standard_model(
                architecture=blackbox_model_config['name'],
                input_shape=config.get('input_shape'),
                n_labels=n_tasks,
                **blackbox_model_config,
            )
            _, latent_name = standard_models.get_out_layer_name_from_config(
                blackbox_model_config["name"],
                add_linear_layers=blackbox_model_config.get('add_linear_layers'),
            )
            embedding_generator = standard_models.create_feature_extractor(
                bbox,
                return_nodes={latent_name: latent_name},
            )
            config['pretrained_model'] = embedding_generator
        run_name = config["run_name"]
        split = config.get("split", 0)
        if split is not None:
            full_run_name = (
                f"{run_name}_fold_{split + 1}"
            )
        else:
            full_run_name = (
                f"{run_name}"
            )
        result_dir = config.get("result_dir", ".")
        cav_path = os.path.join(
            result_dir,
            f'{full_run_name}_CAVs.npy'
        )
        intercept_path = os.path.join(
            result_dir,
            f'{full_run_name}_Intercepts.npy'
        )
        if (config.get('concept_vectors', None) is None) and (
            os.path.exists(cav_path)
        ):
            config['concept_vectors'] = torch.FloatTensor(np.load(cav_path))


        if (config.get('concept_vector_intercepts', None) is None) and (
            os.path.exists(intercept_path)
        ):
            config['concept_vector_intercepts'] = torch.FloatTensor(
                np.load(intercept_path)
            )

        if not config.get('emb_size'):
            config['emb_size'] = config['concept_vectors'].shape[-1]
        return models_pcbm.PCBM(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=config['emb_size'],
            task_class_weights=(
                torch.FloatTensor(task_class_weights)
                if (task_class_weights is not None)
                else None
            ),
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0),
            optimizer=config['optimizer'],
            top_k_accuracy=config.get('top_k_accuracy'),
            output_latent=output_latent,
            output_interventions=output_interventions,
            concept_vectors=config.get('concept_vectors'),
            concept_vector_intercepts=config.get('concept_vector_intercepts'),
            pretrained_model=config['pretrained_model'],
            c2y_model=config.get('c2y_model'),
            residual=config.get('residual', False),
            residual_model=config.get('residual_model', None),
            reg_strength=config.get('reg_strength', 1e-5),
            l1_ratio=config.get('reg_strength', 0.99),
            lr_scheduler_factor=config.get('lr_scheduler_factor', 0.1),
            lr_scheduler_patience=config.get('lr_scheduler_patience', 10),
            freeze_pretrained_model=config.get('freeze_pretrained_model', True),
            freeze_concept_embeddings=config.get('freeze_concept_embeddings', True),
            training_intervention_prob=config.get(
                'training_intervention_prob',
                0.0,
            ),
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
        )

    elif (
        "ConceptBottleneckModel" in config["architecture"] or
        "CBM" in config["architecture"]
    ):
        model_cls = models_cbm.ConceptBottleneckModel
        extra_params = {
            "bool": config["bool"],
            "extra_dims": config["extra_dims"],
            "sigmoidal_extra_capacity": config.get(
                "sigmoidal_extra_capacity",
                True,
            ),
            "sigmoidal_prob": config.get("sigmoidal_prob", True),
            "intervention_policy": intervention_policy,
            "bottleneck_nonlinear": config.get("bottleneck_nonlinear", None),
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "x2c_model": x2c_model,
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
        }


    elif "NewMixingConceptEmbeddingModel" in config["architecture"]:
        if config['architecture'] == 'NeSyMixingConceptEmbeddingModel':
            model_cls = models_mixcem.NeSyMixingConceptEmbeddingModel
        else:
            model_cls = models_mixcem.MixingConceptEmbeddingModel
        extra_params = {
            'rollout_init_steps': config.get('rollout_init_steps', 0),
            'int_model_layers': config.get('int_model_layers', None),
            'int_model_use_bn': config.get('int_model_use_bn', True),
            'num_rollouts': config.get('num_rollouts', 1),
            'intervention_discount': config.get('intervention_discount', 1),
            'include_only_last_trajectory_loss': config.get('include_only_last_trajectory_loss', True),
            'intervention_task_loss_weight': config.get('intervention_task_loss_weight', 1),
            'intervention_weight': config.get('intervention_weight', 5),
            'concept_map': config.get('concept_map', None),
            'max_horizon': config.get('max_horizon', 6),
            'initial_horizon': config.get('initial_horizon', 2),
            'horizon_rate': config.get('horizon_rate', 1.005),


            "emb_size": config["emb_size"],
            "normalize_embs": config.get(
                "normalize_embs",
                False,
            ),
            "intermediate_task_concept_loss": config.get(
                "intermediate_task_concept_loss",
                0.0,
            ),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                1,
            ),
            "mix_ground_truth_embs": config.get(
                "mix_ground_truth_embs",
                True,
            ),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                None,
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "fixed_embeddings": config.get('fixed_embeddings', False),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "use_cosine_similarity": config.get('use_cosine_similarity', False),
            "use_linear_emb_layer": config.get('use_linear_emb_layer', False),
            "fixed_scale": config.get('fixed_scale', None),
            "residual_scale": config.get('residual_scale', 1),
            "conditional_residual": config.get('conditional_residual', False),
            "residual_layers": config.get("residual_layers", []),
            "bottleneck_pooling": config.get('bottleneck_pooling', 'concat'),
            "per_concept_residual": config.get('per_concept_residual', False),
            "shared_per_concept_residual": config.get('shared_per_concept_residual', False),
            "sigmoidal_residual": config.get('sigmoidal_residual', False),
            "residual_deviation": config.get('residual_deviation', False),
            "warmup_mode": train and (config.get('blackbox_warmup_epochs', 0) > 0),
            "include_bypass_model": (config.get('blackbox_warmup_epochs', 0) > 0),
            "residual_norm_loss": config.get('residual_norm_loss', 0),
            "learnable_residual_scale": config.get('learnable_residual_scale', False),
            "sigmoidal_residual_scale": config.get('sigmoidal_residual_scale', False),
            "learn_residual_embeddings": config.get("learn_residual_embeddings", False),
            "residual_norm_metric": config.get('residual_norm_metric', 1),
            "residual_scale_norm_metric": config.get('residual_scale_norm_metric', 1),
            "noise_residual_embedings": config.get('noise_residual_embedings', False),
            "dynamic_residual": config.get('dynamic_residual', False),
            "learnable_distance_metric": config.get('learnable_distance_metric', False),
            "learnable_prob_model": config.get('learnable_prob_model', False),
            "residual_model_weight_l2_reg": config.get('residual_model_weight_l2_reg', 0),
            "extra_capacity_dropout_prob": config.get('extra_capacity_dropout_prob', 0),
            "extra_capacity": config.get("extra_capacity", 0),
            "orthogonal_extra_capacity": config.get("orthogonal_extra_capacity", False),
            "use_residual": config.get('use_residual', True),
        }
        # extra_params = {
        #     "emb_size": config["emb_size"],
        #     "normalize_embs": config.get(
        #         "normalize_embs",
        #         False,
        #     ),
        #     "intermediate_task_concept_loss": config.get(
        #         "intermediate_task_concept_loss",
        #         0.0,
        #     ),
        #     "intervention_task_discount": config.get(
        #         "intervention_task_discount",
        #         1,
        #     ),
        #     "mix_ground_truth_embs": config.get(
        #         "mix_ground_truth_embs",
        #         True,
        #     ),
        #     "intervention_policy": intervention_policy,
        #     "training_intervention_prob": config.get(
        #         'training_intervention_prob',
        #         0.25,
        #     ),
        #     "embedding_activation": config.get(
        #         "embedding_activation",
        #         None,
        #     ),
        #     "c2y_model": c2y_model,
        #     "c2y_layers": config.get("c2y_layers", []),

        #     "fixed_embeddings": config.get('fixed_embeddings', False),
        #     "initial_concept_embeddings": config.get(
        #         'initial_concept_embeddings',
        #         None,
        #     ),
        #     "use_cosine_similarity": config.get('use_cosine_similarity', False),
        #     "use_linear_emb_layer": config.get('use_linear_emb_layer', False),
        #     "fixed_scale": config.get('fixed_scale', None),
        #     "residual_scale": config.get('residual_scale', 1),
        #     "conditional_residual": config.get('conditional_residual', False),
        #     "residual_layers": config.get("residual_layers", []),
        #     "bottleneck_pooling": config.get('bottleneck_pooling', 'concat'),
        #     "per_concept_residual": config.get('per_concept_residual', False),
        #     "shared_per_concept_residual": config.get('shared_per_concept_residual', False),
        #     "sigmoidal_residual": config.get('sigmoidal_residual', False),
        #     "residual_deviation": config.get('residual_deviation', False),
        #     "warmup_mode": train and (config.get('blackbox_warmup_epochs', 0) > 0),
        #     "include_bypass_model": (config.get('blackbox_warmup_epochs', 0) > 0),
        #     "residual_norm_loss": config.get('residual_norm_loss', 0),
        #     "learnable_residual_scale": config.get('learnable_residual_scale', False),
        #     "sigmoidal_residual_scale": config.get('sigmoidal_residual_scale', False),
        #     "learn_residual_embeddings": config.get("learn_residual_embeddings", False),
        #     "residual_norm_metric": config.get('residual_norm_metric', 1),
        #     "residual_scale_norm_metric": config.get('residual_scale_norm_metric', 1),
        #     "noise_residual_embedings": config.get('noise_residual_embedings', False),
        #     "dynamic_residual": config.get('dynamic_residual', False),
        #     "learnable_distance_metric": config.get('learnable_distance_metric', False),
        #     "learnable_prob_model": config.get('learnable_prob_model', False),
        #     "residual_model_weight_l2_reg": config.get('residual_model_weight_l2_reg', 0),
        #     "extra_capacity_dropout_prob": config.get('extra_capacity_dropout_prob', 0),
        #     "extra_capacity": config.get("extra_capacity", 0),
        #     "orthogonal_extra_capacity": config.get("orthogonal_extra_capacity", False),
        #     "use_residual": config.get('use_residual', True),
        # }
    elif config['architecture'] == "ProjectionConceptEmbeddingModel":
        model_cls = direction_cem.ProjectionConceptEmbeddingModel
        extra_params = {
            'rollout_init_steps': config.get('rollout_init_steps', 0),
            'int_model_layers': config.get('int_model_layers', None),
            'int_model_use_bn': config.get('int_model_use_bn', True),
            'num_rollouts': config.get('num_rollouts', 1),
            'intervention_discount': config.get('intervention_discount', 1),
            'include_only_last_trajectory_loss': config.get('include_only_last_trajectory_loss', True),
            'intervention_task_loss_weight': config.get('intervention_task_loss_weight', 1),
            'intervention_weight': config.get('intervention_weight', 5),
            'concept_map': config.get('concept_map', None),
            'max_horizon': config.get('max_horizon', 6),
            'initial_horizon': config.get('initial_horizon', 2),
            'horizon_rate': config.get('horizon_rate', 1.005),



            "emb_size": config["emb_size"],
            "normalize_embs": config.get(
                "normalize_embs",
                False,
            ),
            "intermediate_task_concept_loss": config.get(
                "intermediate_task_concept_loss",
                0.0,
            ),
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                1,
            ),
            "mix_ground_truth_embs": config.get(
                "mix_ground_truth_embs",
                True,
            ),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                None,
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "fixed_embeddings": config.get('fixed_embeddings', False),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "use_cosine_similarity": config.get('use_cosine_similarity', False),
            "use_linear_emb_layer": config.get('use_linear_emb_layer', False),
            "fixed_scale": config.get('fixed_scale', None),
            "residual_scale": config.get('residual_scale', 1),
            "residual_layers": config.get("residual_layers", []),
            "bottleneck_pooling": config.get('bottleneck_pooling', 'concat'),
            "per_concept_residual": config.get('per_concept_residual', False),
            "shared_per_concept_residual": config.get('shared_per_concept_residual', False),
            "sigmoidal_residual": config.get('sigmoidal_residual', False),
            "residual_deviation": config.get('residual_deviation', False),
            "warmup_mode": train and (config.get('blackbox_warmup_epochs', 0) > 0),
            "include_bypass_model": (config.get('blackbox_warmup_epochs', 0) > 0),
            "residual_norm_loss": config.get('residual_norm_loss', 0),
            "learnable_residual_scale": config.get('learnable_residual_scale', False),
            "sigmoidal_residual_scale": config.get('sigmoidal_residual_scale', False),
            "learn_residual_embeddings": config.get("learn_residual_embeddings", False),
            "residual_norm_metric": config.get('residual_norm_metric', 1),
            "residual_scale_norm_metric": config.get('residual_scale_norm_metric', 1),
            "noise_residual_embedings": config.get('noise_residual_embedings', False),
            "dynamic_residual": config.get('dynamic_residual', False),
            "learnable_distance_metric": config.get('learnable_distance_metric', False),
            "learnable_prob_model": config.get('learnable_prob_model', False),
            "residual_model_weight_l2_reg": config.get('residual_model_weight_l2_reg', 0),
            "extra_capacity": config.get("extra_capacity", 0),
            "orthogonal_extra_capacity": config.get("orthogonal_extra_capacity", False),
            "use_residual": config.get('use_residual', True),
            "simplified_mode": config.get('simplified_mode', False),
            "use_triplet_loss": config.get('use_triplet_loss', False),
            "learnable_orthogonal_dir": config.get('learnable_orthogonal_dir', False),
            "single_residual_vector": config.get('single_residual_vector', False),
            "extra_capacity_dropout_prob": config.get('extra_capacity_dropout_prob', 0),

            "adversary_loss_weight": config.get('adversary_loss_weight', 0),
            "use_learnable_residual": config.get('use_learnable_residual', False),
            "warmup_period": config.get('warmup_period', 0),
            "sigmoidal_extra_capacity": config.get('sigmoidal_extra_capacity', True),
            "conditional_residual": config.get('conditional_residual', True),
            "use_learnable_prob": config.get('use_learnable_prob', False),
            "dyn_scaling": config.get('dyn_scaling', 100),

            "residual_weight_l2": config.get('residual_weight_l2', 0),
            "residual_drop_prob": config.get('residual_drop_prob', 0),
            "mix_residuals": config.get('mix_residuals', False),
            "manual_residual_scale": config.get('manual_residual_scale', 1),
            "residual_sep_loss": config.get('residual_sep_loss', 0),
            "residual_ood_detection": config.get('residual_ood_detection', 1),
        }


    elif config['architecture'] == "DeferConceptEmbeddingModel":
        model_cls = defer_cem.DeferConceptEmbeddingModel
        extra_params = {
            'rollout_init_steps': config.get('rollout_init_steps', 0),
            'int_model_layers': config.get('int_model_layers', None),
            'int_model_use_bn': config.get('int_model_use_bn', True),
            'num_rollouts': config.get('num_rollouts', 1),
            'intervention_discount': config.get('intervention_discount', 1),
            'include_only_last_trajectory_loss': config.get('include_only_last_trajectory_loss', True),
            'intervention_task_loss_weight': config.get('intervention_task_loss_weight', 1),
            'intervention_weight': config.get('intervention_weight', 5),
            'concept_map': config.get('concept_map', None),
            'max_horizon': config.get('max_horizon', 6),
            'initial_horizon': config.get('initial_horizon', 2),
            'horizon_rate': config.get('horizon_rate', 1.005),




            "emb_size": config["emb_size"],
            "intervention_task_discount": config.get(
                "intervention_task_discount",
                1,
            ),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.25,
            ),
            "embedding_activation": config.get(
                "embedding_activation",
                None,
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),

            "fixed_embeddings": config.get('fixed_embeddings', False),
            "fixed_scale": config.get('fixed_scale', None),
            "initial_concept_embeddings": config.get(
                'initial_concept_embeddings',
                None,
            ),
            "bottleneck_pooling": config.get('bottleneck_pooling', 'concat'),

            'bottleneck_pooling': config.get('bottleneck_pooling', 'concat'),
            'fixed_embeddings': config.get('fixed_embeddings', False),
            'initial_concept_embeddings': config.get('initial_concept_embeddings', None),
            'fixed_scale': config.get('fixed_scale', None),
            'simplified_mode': config.get('simplified_mode', False),
            'shared_emb_generator': config.get('shared_emb_generator', True),
            'dynamic_ood_detection': config.get('dynamic_ood_detection', 1),
            'dynamic_weights_reg': config.get('dynamic_weights_reg', 0),
            'dynamic_weights_reg_norm': config.get('dynamic_weights_reg_norm', 2),
            'dynamic_activations_reg': config.get('dynamic_activations_reg', 0),
            'dynamic_activations_reg_norm': config.get('dynamic_activations_reg_norm', 2),
            'dynamic_drop_prob': config.get('dynamic_drop_prob', 0),
            'conditional_pred_mixture': config.get('conditional_pred_mixture', False),
            'mode': config.get('mode', 'joint'),
            "mixcem_c2y_scaling": config.get('mixcem_c2y_scaling', 1),
            "dynamic_c2y_scaling": config.get('dynamic_c2y_scaling', 1),
            "dyn_from_shared_space": config.get('dyn_from_shared_space', True),
            "probability_mode": config.get('probability_mode', 'joint'),
            "mix_before_predictor": config.get('mix_before_predictor', False),
            "joint_model_pooling": config.get('joint_model_pooling', None),
        }

    else:
        raise ValueError(f'Invalid architecture "{config["architecture"]}"')

    if isinstance(config["c_extractor_arch"], str):
        if config["c_extractor_arch"] == "resnet18":
            c_extractor_arch = resnet18
        elif config["c_extractor_arch"] == "resnet34":
            c_extractor_arch = resnet34
        elif config["c_extractor_arch"] == "resnet50":
            c_extractor_arch = resnet50
        elif config["c_extractor_arch"] == "densenet121":
            c_extractor_arch = densenet121
        elif  config["c_extractor_arch"] == "identity":
            c_extractor_arch = "identity"
        else:
            raise ValueError(f'Invalid model_to_use "{config["model_to_use"]}"')
    else:
        c_extractor_arch = config["c_extractor_arch"]

    # Create model
    return model_cls(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        weight_loss=(
            torch.FloatTensor(imbalance)
            if config['weight_loss'] and (imbalance is not None)
            else None
        ),
        task_class_weights=(
            torch.FloatTensor(task_class_weights)
            if (task_class_weights is not None)
            else None
        ),
        concept_loss_weight=config['concept_loss_weight'],
        task_loss_weight=task_loss_weight,
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0),
        c_extractor_arch=utils.wrap_pretrained_model(c_extractor_arch),
        optimizer=config['optimizer'],
        lr_scheduler_factor=config.get('lr_scheduler_factor', 0.1),
        lr_scheduler_patience=config.get('lr_scheduler_patience', 10),
        top_k_accuracy=config.get('top_k_accuracy'),
        output_latent=output_latent,
        output_interventions=output_interventions,
        **extra_params,
    )


def construct_sequential_models(
    n_concepts,
    n_tasks,
    config,
    imbalance=None,
    task_class_weights=None,
):
    assert config.get('extra_dims', 0) == 0, (
        "We can only train sequential/joint models if the extra "
        "dimensions are 0!"
    )
    if isinstance(config["c_extractor_arch"], str):
        if config["c_extractor_arch"] == "resnet18":
            c_extractor_arch = resnet18
        elif config["c_extractor_arch"] == "resnet34":
            c_extractor_arch = resnet34
        elif config["c_extractor_arch"] == "resnet50":
            c_extractor_arch = resnet50
        elif config["c_extractor_arch"] == "densenet121":
            c_extractor_arch = densenet121
        else:
            raise ValueError(
                f'Invalid model_to_use "{config["model_to_use"]}"'
            )
    else:
        c_extractor_arch = config["c_extractor_arch"]
    # Else we assume that it is a callable function which we will
    # need to instantiate here
    try:
        x2c_model = c_extractor_arch(
            pretrained=config.get('pretrain_model', True),
        )
        if c_extractor_arch == densenet121:
            x2c_model.classifier = torch.nn.Linear(1024, n_concepts)
        elif hasattr(x2c_model, 'fc'):
            x2c_model.fc = torch.nn.Linear(512, n_concepts)
    except Exception as e:
        x2c_model = c_extractor_arch(output_dim=n_concepts)
    x2c_model = utils.WrapperModule(
        n_tasks=n_concepts,
        model=x2c_model,
        weight_loss=(
            torch.FloatTensor(imbalance)
            if config['weight_loss'] and (imbalance is not None)
            else None
        ),
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0),
        optimizer=config['optimizer'],
        lr_scheduler_factor=config.get('lr_scheduler_factor', 0.1),
        lr_scheduler_patience=config.get('lr_scheduler_patience', 10),
        binary_output=True,
        sigmoidal_output=True,
    )

    # Now construct the label prediction model
    # Else we construct it here directly
    c2y_layers = config.get('c2y_layers', [])
    units = [n_concepts] + (c2y_layers or []) + [n_tasks]
    layers = [
        torch.nn.Linear(units[i-1], units[i])
        for i in range(1, len(units))
    ]
    c2y_model = utils.WrapperModule(
        n_tasks=n_tasks,
        model=torch.nn.Sequential(*layers),
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0),
        optimizer=config['optimizer'],
        lr_scheduler_factor=config.get('lr_scheduler_factor', 0.1),
        lr_scheduler_patience=config.get('lr_scheduler_patience', 10),
        top_k_accuracy=config.get('top_k_accuracy'),
        binary_output=False,
        sigmoidal_output=False,
        weight_loss=(
            torch.FloatTensor(task_class_weights)
            if (task_class_weights is not None)
            else None
        ),
    )
    return x2c_model, c2y_model


################################################################################
## MODEL LOADING
################################################################################


def load_trained_model(
    config,
    n_tasks,
    result_dir,
    n_concepts,
    split=0,
    imbalance=None,
    task_class_weights=None,
    train_dl=None,
    logger=False,
    accelerator="auto",
    devices="auto",
    intervention_policy=None,
    intervene=False,
    output_latent=False,
    output_interventions=False,
    enable_checkpointing=False,
):
    if "run_name" in config:
        run_name = config["run_name"]
    else:
        run_name = (
            f"{config['architecture']}{config.get('extra_name', '')}"
        )
    if split is not None:
        full_run_name = (
            f"{run_name}_fold_{split + 1}"
        )
    else:
        full_run_name = run_name
    independent = False
    sequential = False
    if config['architecture'].startswith("Sequential"):
        sequential = True
    elif config['architecture'].startswith("Independent"):
        independent = True
    model_saved_path = os.path.join(
        result_dir or ".",
        f'{full_run_name}.pt'
    )

    if (
        ((intervention_policy is not None) or intervene) and
        (train_dl is not None) and (
            (
                (config['architecture'] in ["ConceptBottleneckModel", "CBM"]) and
                (not config.get('sigmoidal_prob', True))
            ) or
            (config['architecture'] in ["PosthocConceptBottleneckModel", "PCBM", "PosthocCBM"])
        )
    ):
        # Then let's look at the empirical distribution of the logits in order
        # to be able to intervene
        model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )
        model.load_state_dict(torch.load(model_saved_path))
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            enable_checkpointing=enable_checkpointing,
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
                np.percentile(out_embs[:, idx], config.get('active_top_percentile', 95))
            )
            inactive_intervention_values.append(
                np.percentile(out_embs[:, idx], config.get('bottom_top_percentile', 5))
            )

        active_intervention_values = torch.FloatTensor(
            active_intervention_values
        )
        inactive_intervention_values = torch.FloatTensor(
            inactive_intervention_values
        )
    else:
        active_intervention_values = inactive_intervention_values = None
    if independent or sequential:
        _, c2y_model = construct_sequential_models(
            n_concepts,
            n_tasks,
            config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
        )


        # As well as the wrapper CBM model we will use for serialization
        # and testing
        # We will be a bit cheeky and use the model with the task loss
        # weight set to 0 for training with the same dataset
        model_config = copy.deepcopy(config)
        model_config['concept_loss_weight'] = 1
        model_config['task_loss_weight'] = 0
        base_model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=model_config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )
        model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=model_config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_latent=output_latent,
            output_interventions=output_interventions,
            x2c_model=base_model.x2c_model,
            c2y_model=c2y_model,
        )


    else:
        model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )

    model.load_state_dict(torch.load(model_saved_path))
    return model
