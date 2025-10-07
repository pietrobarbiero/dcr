"""
usage: collect_results.py [-h] [--config config.yaml] [--output_dir path] [--collect new_out_config.yaml] [-d] [-p param_name value] [--filter_out regex] [--filter_in regex]
                          [--filter_in_file model_selection_file.joblib] [--extra_datasets_filter_in_file model_selection_file.joblib] [--only_previously_selected] [--model_selection_metrics metric_name]
                          [--ignore_groups group_name] [--summary_table_metrics metric_name pretty_name metric_name pretty_name] [-m group_pattern_regex group_name group_pattern_regex group_name] [--use_auc]
                          [--forced] [--use_selected_names]

Collects the results of experiments ran on a specific config and generates a new config file that include only the runs selected after all experiments have been ran. This is also a useful tool to easily and
quickly print out the results of an already ran set of experiments.

optional arguments:
  -h, --help            show this help message and exit
  --config config.yaml, -c config.yaml
                        YAML file with the configuration for the set of experiments to summarise/collect.
  --output_dir path, -o path
                        directory where we will dump our experiment's results.
  --collect new_out_config.yaml
                        name of the new YAML config file to be used to dump the collected results. If not given, then no file will be generated and only a summary will be printed.
  -d, --debug           starts debug mode in our program.
  -p param_name value, --param param_name value
                        Allows the passing of a config param that will overwrite anything passed as part of the config file itself.
  --filter_out regex    skips runs whose names match the regexes provided via this argument. These regexes must follow Python's regex syntax.
  --filter_in regex     includes only runs whose names match the regexes provided with this argument. These regexes must follow Python's regex syntax.
  --filter_in_file model_selection_file.joblib
                        includes only runs whose names are in the joblib file outputed from a previous model selection run.
  --extra_datasets_filter_in_file model_selection_file.joblib
                        includes for extra dataset evaluation only runs whose names are in the joblib file outputed from a previous model selection run.
  --only_previously_selected
                        it runs the models that were only previously selected by the model selection ran on a previous iteration of this experiment
  --model_selection_metrics metric_name
                        metrics to be used to make a summary table by selecting models based on some (validation) metric. If provided, the one must also provide groups via the model_selection_groups argument.
  --ignore_groups group_name
                        groups to be ignored when we are doing the collection.
  --summary_table_metrics metric_name pretty_name metric_name pretty_name
                        List of metrics to be included as part of the final summary table of this run.
  -m group_pattern_regex group_name group_pattern_regex group_name, --model_selection_groups group_pattern_regex group_name group_pattern_regex group_name
                        Performs model selection based on the requested model selection metrics by grouping methods that match the Python regex `group_pattern_regex` into a single group with name `group_name`.
  --use_auc             use ROC-AUC as the main performance metric rather than accuracy.
  --forced, -f          ignores runs whose results we cannot currently find.
  --use_selected_names, -s
                        we will use the group selection name for each run in the collected config.
"""
import argparse
import copy
import joblib
import logging
import os
import re
import sys
import torch
import yaml



torch.multiprocessing.set_sharing_strategy('file_system')
from collections import defaultdict
from datetime import datetime
from pytorch_lightning import seed_everything


import cem.train.utils as utils
import experiments.experiment_utils as experiment_utils

from run_experiments import _perform_model_selection

################################################################################
## HELPER FUNCTIONS
################################################################################

FIRST_SHARED_KEYS = [
    'results_dir',
    'trials',
    'model_selection_trials',
    'model_selection_groups',
    'model_selection_metrics',
    'dataset_config',
    'intervention_config',
    'eval_config',
    'skip_repr_evaluation',
]

FIRST_RUN_KEYS = [
    'architecture',
    'run_name',
]

class FlowList(list):
    pass

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

def flow_list_representer(dumper, data):
    return dumper.represent_sequence(
        'tag:yaml.org,2002:seq',
        data,
        flow_style=True,
    )

def make_inline_lists(d, max_allowed=5, ignore_keys=None, current_key="", ignore_top_level=False):
    ignore_keys = ignore_keys or []
    if not isinstance(d, (dict, list)):
        # Then we are done!
        return d
    if isinstance(d, list):
        if ignore_top_level or (len(d) > max_allowed):
            return [
                make_inline_lists(
                    d=x,
                    max_allowed=max_allowed,
                    ignore_keys=ignore_keys,
                    ignore_top_level=False,
                )
                for x in d
            ]
        else:
            return FlowList([
                make_inline_lists(
                    d=x,
                    max_allowed=max_allowed,
                    ignore_keys=ignore_keys,
                    ignore_top_level=False,
                )
                for x in d
            ])

    for key, val in d.items():
        inner_result = make_inline_lists(
            d=val,
            max_allowed=max_allowed,
            ignore_keys=ignore_keys,
            ignore_top_level=(key in ignore_keys),
        )
        if isinstance(inner_result, list) and (len(inner_result) <= max_allowed) and (
            key not in ignore_keys
        ):
            # Then let's make it a in-line list
            d[key] = FlowList(inner_result)
        else:
            d[key] = inner_result
    return d


class QuotedString(str):
    pass

def quote_strings(obj):
    if isinstance(obj, str):
        return QuotedString(obj)
    elif isinstance(obj, list):
        return [quote_strings(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: quote_strings(v) for k, v in obj.items()}
    else:
        return obj


def quoted_scalar_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style='"')

yaml.add_representer(QuotedString, quoted_scalar_representer)
yaml.add_representer(QuotedString, quoted_scalar_representer, Dumper=IndentDumper)


yaml.add_representer(FlowList, flow_list_representer)
yaml.add_representer(FlowList, flow_list_representer, Dumper=IndentDumper)


################################################################################
## MAIN FUNCTION
################################################################################


def construct_selected_config(
    result_dir,
    experiment_config,
    global_params=None,
    summary_table_metrics=None,
    sort_key="Task Accuracy",
    filter_out_regex=None,
    filter_in_regex=None,
    model_selection_metrics=None,
    model_selection_groups=None,
    use_auc=False,
    forced=False,
    ignore_groups=None,
    use_selected_names=False,
):
    ignore_groups = set(ignore_groups or [])
    seed_everything(42)
    # parameters for data, model, and training
    experiment_config = copy.deepcopy(experiment_config)
    if 'shared_params' not in experiment_config:
        experiment_config['shared_params'] = {}
    # Move all global things into the shared params
    shared_params = experiment_config['shared_params']
    for key, vals in experiment_config.items():
        if key not in ['runs', 'shared_params']:
            shared_params[key] = vals

    utils.extend_with_global_params(
        shared_params, global_params or []
    )

    # Set log level in env variable as this will be necessary for
    # subprocessing
    os.environ['LOGLEVEL'] = os.environ.get(
        'LOGLEVEL',
        logging.getLevelName(logging.getLogger().getEffectiveLevel()),
    )
    loglevel = os.environ['LOGLEVEL']
    logging.info(f'Setting log level to: "{loglevel}"')

    # Things regarding model selection
    model_selection_trials = shared_params.get(
        'model_selection_trials',
        shared_params["trials"],
    )
    models_selected_to_continue = None
    included_models = None

    os.makedirs(result_dir, exist_ok=True)
    results = {}

    # We will keep a map contaning the config used for each run
    run_to_config = {}
    for split in range(
        shared_params.get("start_split", 0),
        shared_params["trials"],
    ):
        results[f'{split}'] = defaultdict(dict)
        now = datetime.now()
        print(
            f"[TRIAL "
            f"{split + 1}/{shared_params['trials']} "
            f"BEGINS AT {now.strftime('%d/%m/%Y at %H:%M:%S')}"
        )
        # And then over all runs in a given trial
        runs = experiment_config['runs']
        for current_config in runs:
            # Construct the config for this particular trial
            trial_config = copy.deepcopy(shared_params)
            trial_config.update(current_config)
            # Time to try as many seeds as requested
            for run_config in experiment_utils.generate_hyperparameter_configs(
                trial_config
            ):
                run_config = copy.deepcopy(run_config)
                run_config['result_dir'] = result_dir
                run_config['split'] = split
                experiment_utils.evaluate_expressions(run_config, soft=True)

                old_results = {}
                if "run_name" not in run_config:
                    run_name = (
                        f"{run_config['architecture']}"
                        f"{run_config.get('extra_name', '')}"
                    )
                    logging.warning(
                        f'Did not find a run name so using the '
                        f'name "{run_name}" by default'
                    )
                    run_config["run_name"] = run_name
                run_name = run_config["run_name"]

                # Determine filtering in and filtering out of run
                if filter_out_regex:
                    skip = False
                    for reg in filter_out_regex:
                        if re.search(reg, f'{run_name}_split_{split}'):
                            logging.info(
                                f'Skipping run '
                                f'{f"{run_name}_split_{split}"} as it '
                                f'matched filter-out regex {reg}'
                            )
                            skip = True
                            break
                    if skip:
                        continue
                if filter_in_regex:
                    found = False
                    for reg in filter_in_regex:
                        if re.search(reg, f'{run_name}_split_{split}'):
                            found = True
                            logging.info(
                                f'Including run '
                                f'{f"{run_name}_split_{split}"} as it '
                                f'did matched filter-in regex {reg}'
                            )
                            break
                    if not found:
                        logging.info(
                            f'Skipping run {f"{run_name}_split_{split}"} as it '
                            f'did not match any filter-in regexes'
                        )
                        continue

                if models_selected_to_continue and (
                    run_name not in models_selected_to_continue
                ):
                    logging.info(
                        f'Skipping run {f"{run_name}_split_{split}"} it was '
                        f'not selected based on any of the model-selection '
                        f'metrics measured after fold {model_selection_trials}'
                    )
                    continue

                # Determine training rerun or not
                current_results_path = os.path.join(
                    result_dir,
                    f'{run_name}_split_{split}_results.joblib'
                )

                found_old_file = os.path.exists(current_results_path)
                if found_old_file:
                    try:
                        with open(current_results_path, 'rb') as f:
                            old_results = joblib.load(f)
                    except Exception as e:
                        logging.info(
                            f'\t\t[IMPORTANT] We found previous results for '
                            f'run {run_name} at trial {split + 1} but we were '
                            f'unable to properly open them after encountering '
                            f'exception {e}'
                        )
                        found_old_file = False

                if (not forced) and (not found_old_file):
                    raise ValueError(
                        f'We could not find the results corresponding to '
                        f'"{run_name}" for split {split}. If you want this to '
                        f'be ignored, please run this with the --force flag on.'
                    )


                results[f'{split}'][run_name].update(old_results)
                run_to_config[run_name] = run_config


        if (model_selection_groups is not None) and (
            model_selection_metrics is not None
        ) and (
            models_selected_to_continue is None
        ) and (
            (split + 1) >= model_selection_trials
        ):
            # Then time to do model selection to avoid running every
            # iteration on every seed
            model_selection_results = _perform_model_selection(
                model_selection_groups=model_selection_groups,
                model_selection_metrics=model_selection_metrics,
                results=results,
                result_dir=result_dir,
                split=split,
                summary_table_metrics=summary_table_metrics,
                config=experiment_config,
                use_auc=use_auc,
            )
            models_selected_to_continue = set()
            included_models = []
            for _, selection_map, _ in model_selection_results:
                included_models.append(set())
                for _, group_selected_method in selection_map.items():
                    included_models[-1].add(group_selected_method)
                    models_selected_to_continue.add(group_selected_method)

            logging.debug(f"\t\tDone with trial {split + 1}")
    print(f"********** End Experiment Results **********")
    experiment_utils.print_table(
        config=experiment_config,
        results=results,
        summary_table_metrics=summary_table_metrics,
        sort_key=sort_key,
        result_dir=result_dir,
        split=split,
        use_auc=(use_auc or run_config.get("n_tasks", 3) <= 2),
        use_int_auc=use_auc or run_config.get('intervention_config', {}).get(
            'use_auc',
            False,
        ),
    )

    # Perform the final model selection
    model_selection_results = _perform_model_selection(
        model_selection_groups=model_selection_groups,
        model_selection_metrics=model_selection_metrics,
        results=results,
        result_dir=result_dir,
        split=split,
        summary_table_metrics=summary_table_metrics,
        config=experiment_config,
        included_models=included_models,
        use_auc=use_auc,
    )

    # And time to wrap up the selected methods into a single large config:
    output_config = {}
    output_config['shared_params'] = {}
    # Order the resulting shared parameters so that we always have
    # the result directory and number of trials on the top
    for key in FIRST_SHARED_KEYS:
        if key in experiment_config['shared_params']:
            output_config['shared_params'][key] = copy.deepcopy(
                experiment_config['shared_params'][key]
            )
    # Now add the rest of keys
    for key, val in experiment_config['shared_params'].items():
        if key in output_config['shared_params']:
            continue
        output_config['shared_params'][key] = copy.deepcopy(val)

    output_config['shared_params'].pop('grid_variables', None)
    output_config['shared_params'].pop('grid_search_mode', None)
    output_config['shared_params'].pop('model_selection_metrics', None)
    output_config['shared_params'].pop('model_selection_groups', None)
    output_config['runs'] = []

    # Time to see which runs we will include here!
    runs_to_include = set()
    for _, selection_map, model_selection_metric in model_selection_results:
        print(
            "*"*15,
            f"Methods selected for metric {model_selection_metric}",
            "*"*15,
        )
        for selected_name, group_selected_method in selection_map.items():
            if selected_name in ignore_groups:
                # Then we will ignore this group!
                continue
            print(f"\t{selected_name} -> {group_selected_method}")
            if group_selected_method not in runs_to_include:
                runs_to_include.add(group_selected_method)
                copied_run_config = {}
                selected_config = run_to_config[group_selected_method]
                for key in FIRST_RUN_KEYS:
                    if key in selected_config:
                        copied_run_config[key] = copy.deepcopy(
                            selected_config[key]
                        )
                for key, val in selected_config.items():
                    if key not in copied_run_config:
                        copied_run_config[key] = copy.deepcopy(val)
                if use_selected_names:
                    copied_run_config['run_name'] = selected_name
                output_config['runs'].append(copied_run_config)

    # Finally, let's try and cleanup the config by removing parameters from
    # runs that are already set to those values in the shared parameters.
    iteration_config = copy.deepcopy(output_config)
    shared_params = iteration_config['shared_params']


    # Make sure some of the keys don't look awkward
    for og_run, current_run in zip(
        output_config['runs'],
        iteration_config['runs'],
    ):
        for run_key, run_val in current_run.items():
            # We will be naive and lazy here and only consider strict match
            # as a reason for removal (even if the result config may be
            # simpler when only a few keys inside a field have different values)
            if run_key in shared_params and shared_params[run_key] == run_val:
                og_run.pop(run_key)
        # And get rid of some superflous keys
        og_run.pop('split', None)
        og_run.pop('result_dir', None)
        og_run.pop('grid_variables', None)
        og_run.pop('grid_search_mode', None)
        og_run.pop('model_selection_metrics', None)
        og_run.pop('model_selection_groups', None)

    output_config = quote_strings(output_config)
    make_inline_lists(
        d=output_config,
        max_allowed=5,
        ignore_keys=[
            'runs',
            'grid_variables',
            'model_selection_metrics',
            'val_intervention_policies',
            'intervention_policies',
            'additional_metrics',
            'additional_test_sets',
        ],
    )
    return output_config


################################################################################
## Arg Parser
################################################################################


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Collects the results of experiments ran on a specific config and '
            'generates a new config file that include only the runs selected '
            'after all experiments have been ran. This is also a useful tool '
            'to easily and quickly print out the results of an already ran '
            'set of experiments.'
        ),
    )
    parser.add_argument(
        '--config',
        '-c',
        help=(
            "YAML file with the configuration for the set of experiments to "
            "summarise/collect."
        ),
        metavar="config.yaml",
    )

    parser.add_argument(
        '--output_dir',
        '-o',
        default=None,
        help=(
            "directory where we will dump our experiment's results."
        ),
        metavar="path",
    )

    parser.add_argument(
        '--collect',
        default=None,
        help=(
            "name of the new YAML config file to be used to dump the collected "
            "results. If not given, then no file will be generated and only a "
            "summary will be printed."
        ),
        metavar="new_out_config.yaml",

    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )
    parser.add_argument(
        '-p',
        '--param',
        action='append',
        nargs=2,
        metavar=('param_name', 'value'),
        help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
        ),
        default=[],
    )
    parser.add_argument(
        "--filter_out",
        action='append',
        metavar=('regex'),
        default=None,
        help=(
            "skips runs whose names match the regexes provided via this "
            "argument. These regexes must follow Python's regex syntax."
        ),
    )
    parser.add_argument(
        "--filter_in",
        action='append',
        metavar=('regex'),
        default=None,
        help=(
            "includes only runs whose names match the regexes provided with "
            "this argument. These regexes must follow Python's regex syntax."
        ),
    )
    parser.add_argument(
        "--filter_in_file",
        action='append',
        metavar=('model_selection_file.joblib'),
        default=None,
        help=(
            "includes only runs whose names are in the joblib file outputed "
            "from a previous model selection run."
        ),
    )
    parser.add_argument(
        "--extra_datasets_filter_in_file",
        action='append',
        metavar=('model_selection_file.joblib'),
        default=None,
        help=(
            "includes for extra dataset evaluation only runs whose names are "
            "in the joblib file outputed from a previous model selection run."
        ),
    )
    parser.add_argument(
        "--only_previously_selected",
        action="store_true",
        default=False,
        help=(
            "it runs the models that were only previously selected by the "
            "model selection ran on a previous iteration of this experiment"
        ),
    )
    parser.add_argument(
        "--model_selection_metrics",
        action='append',
        metavar=('metric_name'),
        default=None,
        help=(
            "metrics to be used to make a summary table by selecting models "
            "based on some (validation) metric. If provided, the one must "
            "also provide groups via the model_selection_groups argument."
        ),
    )
    parser.add_argument(
        "--ignore_groups",
        action='append',
        metavar=('group_name'),
        default=None,
        help=(
            "groups to be ignored when we are doing the collection."
        ),
    )
    parser.add_argument(
        "--summary_table_metrics",
        action='append',
        nargs=2,
        metavar=('metric_name pretty_name'),
        help=(
            'List of metrics to be included as part of the final summary '
            'table of this run.'
        ),
        default=None,
    )

    parser.add_argument(
        "-m",
        "--model_selection_groups",
        action='append',
        nargs=2,
        metavar=('group_pattern_regex group_name'),
        help=(
            'Performs model selection based on the requested model selection '
            'metrics by grouping methods that match the Python regex '
            '`group_pattern_regex` into a single group with name '
            '`group_name`.'
        ),
        default=[],
    )
    parser.add_argument(
        '--use_auc',
        action="store_true",
         default=False,
         help=(
             "use ROC-AUC as the main performance metric rather than accuracy."
         ),
    )
    parser.add_argument(
        '--forced',
        '-f',
        action="store_true",
         default=False,
         help=(
            "ignores runs whose results we cannot currently find."
         ),
    )
    parser.add_argument(
        '--use_selected_names',
        '-s',
        action="store_true",
         default=False,
         help=(
            "we will use the group selection name for each run in the "
            "collected config."
         ),
    )
    return parser


################################################################################
## Main Entry Point
################################################################################

if __name__ == '__main__':
    # Build our arg parser first
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    if args.config:
        with open(args.config, "r") as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        loaded_config = {}
    if "shared_params" not in loaded_config:
        loaded_config["shared_params"] = {}
    if "runs" not in loaded_config:
        loaded_config["runs"] = []

    # Finally, time to actually call our main function!
    model_selection_groups = loaded_config.get("model_selection_groups", None)
    if args.model_selection_groups:
        model_selection_groups = args.model_selection_groups

    summary_table_metrics = loaded_config.get("summary_table_metrics", None)
    if args.summary_table_metrics:
        summary_table_metrics = args.summary_table_metrics

    model_selection_metrics = loaded_config.get("model_selection_metrics", None)
    if args.model_selection_metrics:
        model_selection_metrics = args.model_selection_metrics

    result_dir = (
        args.output_dir if args.output_dir
        else loaded_config.get('results_dir', 'results')
    )

    given_filter_in_file = args.filter_in_file
    if args.only_previously_selected and model_selection_metrics:
        if given_filter_in_file is None:
            given_filter_in_file = []
        for model_selection_metric in model_selection_metrics:
            given_filter_in_file.append(
                os.path.join(
                    result_dir,
                    f'selected_models_{model_selection_metric}.joblib'
                )
            )

    if given_filter_in_file is not None:
        if args.filter_in is None:
            args.filter_in = []
        for file_path in given_filter_in_file:
            if not os.path.exists(file_path):
                raise ValueError(
                    f'Path for filter-in file {file_path} is not a valid path'
                )
            loaded_selection = joblib.load(file_path)
            for _, method_name in loaded_selection.items():
                args.filter_in.append(method_name)

    extra_datasets_filter_in_file = None
    if args.extra_datasets_filter_in_file is not None:
        extra_datasets_filter_in_file = []
        if args.filter_in is None:
            args.filter_in = []
        for file_path in args.extra_datasets_filter_in_file:
            if not os.path.exists(file_path):
                raise ValueError(
                    f'Path for extra dataset filter-in file {file_path} is '
                    f'not a valid path'
                )
            loaded_selection = joblib.load(file_path)
            for _, method_name in loaded_selection.items():
                extra_datasets_filter_in_file.append(method_name)

    result_dir = (
        args.output_dir if args.output_dir
        else loaded_config.get('results_dir', 'results')
    )
    out_config = construct_selected_config(
        result_dir=result_dir,
        experiment_config=loaded_config,
        global_params=args.param,
        filter_out_regex=args.filter_out,
        filter_in_regex=args.filter_in,
        model_selection_metrics=model_selection_metrics,
        model_selection_groups=model_selection_groups,
        summary_table_metrics=summary_table_metrics,
        use_auc=args.use_auc,
        forced=args.forced,
        ignore_groups=args.ignore_groups,
        use_selected_names=args.use_selected_names,
    )

    # Save summarised config if reasonable!
    if args.collect:
        with open(args.collect, "w") as f:
            out = yaml.dump(
                out_config,
                sort_keys=False,
                indent=2,
                Dumper=IndentDumper,
            )
            out = out.replace("  - architecture:", "\n  - architecture:")
            for o, n in [
                ('    - val_acc_y\n', ''),
                ('    - val_mixcem_sel_acc\n', ''),
                ('    - [.*(MixCEM_).*(_ce_0).*$, MixCEM Final No Calibration (Baseline)]\n', ''),
                ('    - [.*(MixCEM_).*(_ce_30).*$, MixCEM Final (Baseline)]\n', ''),
                ('    - [.*(MixCEM_).*$, MixCEM Final All (Baseline)]\n', '    - [.*(MixCEM_).*$, MixCEM]\n'),
                (' (Baseline)', ''),
                ('MixCEM Final All', 'MixCEM'),
            ]:
                out = out.replace(o, n)
            f.write(out)

    # And that's it!
    sys.exit(0)
