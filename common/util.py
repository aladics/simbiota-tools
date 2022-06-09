from pathlib import Path
import sys
from typing import List

import yaml

from cdf.function.function import get as get_cdf_value

CONFIG_PATH = "common/config.yaml"


def get_config():
    with Path(CONFIG_PATH).open("r") as f:
        conf = yaml.safe_load(f)
    return conf


CONFIG = get_config()


def reload_config():
    global CONFIG
    CONFIG = get_config()


def get_sha_path_from_result(result_path_str: str) -> str:
    return result_path_str.replace(
        Path(result_path_str).stem, Path(result_path_str).stem + "_sha"
    )


def get_resolved_path(path_str) -> str:
    return str(Path(path_str).resolve())


def set_dbh_path() -> None:
    dbh_path_str = CONFIG["dbh_path"]

    if not Path(dbh_path_str).exists():
        raise ValueError(f"Invalid DBH Path, set it accordingly in {CONFIG_PATH}")

    sys.path.append(str(Path(dbh_path_str).resolve()))


def get_eval_progress_path() -> Path:
    return Path(CONFIG["eval_dir"]) / CONFIG["eval_progress_json"]


def get_eval_input_path() -> Path:
    return Path(CONFIG["eval_dir"]) / "inputs"


def get_eval_results_path() -> Path:
    return Path(CONFIG["eval_dir"]) / "results"


def get_hyper_results_path() -> Path:
    return Path(CONFIG["hyper_results_dir"]) / "results.json"


def generate_task_id(architecture: str, n_run: int, n_week: int) -> str:
    pattern: str = CONFIG["task_id_pattern"]

    return (
        pattern.replace("<architecture>", architecture)
        .replace("<n_run>", str(n_run))
        .replace("<n_week>", str(n_week))
    )


def get_usecase_weights(usecase: str):
    return CONFIG["usecases"][usecase]


def get_usecase_names() -> List[str]:
    return CONFIG["usecases"].keys()


def get_model_names() -> List[str]:
    return CONFIG["model_names"]


def get_weeks_range():
    return range(CONFIG["start_week"], CONFIG["end_week"] + 1)


def get_runs_range():
    return range(CONFIG["start_run"], CONFIG["end_run"] + 1)


def get_arch_types():
    return CONFIG["architecture_types"]


def get_eval_ranks_path() -> Path:
    return Path(CONFIG["eval_dir"]) / "ranks"


def get_metric_names() -> List[str]:
    return CONFIG["metric_names"]


def get_ranks_run_dir(n_run: int, arch: str) -> Path:
    return get_eval_ranks_path() / str(n_run) / arch


def get_ranks_data_path() -> Path:
    return get_eval_ranks_path() / "ranks_data.json"


def get_feature_num() -> int:
    return CONFIG["feature_num"]


def is_standardize() -> bool:
    return CONFIG["standardize"]


def get_standardization() -> str:
    if is_standardize():
        return "--standardize"
    return "--dont-standardize"


def get_deliverable_dir() -> Path:
    return Path(CONFIG["deliverable_dir"])


def get_feature_set_name() -> str:
    return CONFIG["feature_set_name"]


def get_u_score(metaparam_vals, parsed_result, model_name: str):
    """Get the utility score for a model result.

    Parameters
    ----------
    metaparam_vals : dict[str, float]
        The dictionary containing the alpha, beta, gamma, delta parameters
    parsed_result : dict[Any, Any]
        The dictionary containing a specific model's results.
    model_name : str
        The name of the model that the parsed result correspond to.

    Returns
    -------
    tuple[str, float]
        A tuple of the model parameters and the model's utility score (a value in the [0,1] interval)
    """

    prec = metaparam_vals["alpha"] * parsed_result["test"]["precision"]
    recall = metaparam_vals["beta"] * parsed_result["test"]["recall"]

    # Linear model has no hyperparameters, so the CDFs for both time and memory is always 1
    if model_name == "linear":
        memory = metaparam_vals["gamma"] * 1
        time = metaparam_vals["delta"] * 1
        return parsed_result["strategy"], prec + recall + memory + time

    cdf_func_params = [
        "--cache",
        get_resolved_path(CONFIG["cdf_func_cache_dir"]),
        "--source",
        get_resolved_path(CONFIG["cdf_results_dir"]),
        "--model",
        model_name,
        "--metric",
        "memory",
        "--value",
        parsed_result["memory"],
    ]
    memory = metaparam_vals["gamma"] * get_cdf_value(
        cdf_func_params, standalone_mode=False
    )

    cdf_func_params[7] = "time"
    cdf_func_params[9] = parsed_result["prediction time"]
    time = metaparam_vals["delta"] * get_cdf_value(
        cdf_func_params, standalone_mode=False
    )

    return parsed_result["strategy"], prec + recall + memory + time


def get_shared_params():
    shared_params = {
        "label": "label",
        "resample": "none",
        "resample_amount": 0,
        "seed": 1337,
        "output": "output",
        "clean": False,
        "calc_completeness": True,
        "preprocess": [
            # [
            #    'features',
            #    'standardize'
            # ],
            ["labels", "binarize"]
        ],
        "return_results": True,
    }

    if is_standardize():
        shared_params["preprocess"].append(["features", "standardize"])
