from pathlib import Path
import sys

import yaml

CONFIG_PATH = "common/config.yaml"

def get_config():
    with Path(CONFIG_PATH).open("r") as f:
        conf  = yaml.safe_load(f)
    return conf


CONFIG = get_config()

def reload_config():
    global CONFIG
    CONFIG = get_config()

def get_sha_path_from_result(result_path_str : str):
    return result_path_str.replace(Path(result_path_str).stem, Path(result_path_str).stem + "_sha")

def get_resolved_path(path_str):
    return str(Path(path_str).resolve())

def set_dbh_path():
    dbh_path_str = CONFIG['dbh_path']
    
    if not Path(dbh_path_str).exists():
        raise ValueError(f"Invalid DBH Path, set it accordingly in {CONFIG_PATH}")
    
    sys.path.append(str(Path(dbh_path_str).resolve()))
    
def get_eval_progress_path():
    return Path(CONFIG['eval_dir']) / CONFIG['eval_progress_json']

def get_eval_input_path():
    return Path(CONFIG['eval_dir']) / "inputs"

def get_eval_results_path():
    return Path(CONFIG['eval_dir']) / "results"

def get_hyper_results_path():
    return Path(CONFIG['hyper_results_dir']) / 'results.json'

def generate_task_id(architecture : str, n_run : int, n_week : int):
    pattern : str = CONFIG['task_id_pattern']

    return pattern.replace("<architecture>", architecture).replace("<n_run>", str(n_run)).replace("<n_week>", str(n_week)) 