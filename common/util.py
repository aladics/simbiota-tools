from pathlib import Path

import yaml

CONFIG_PATH = "common/config.yaml"

def get_config():
    with Path(CONFIG_PATH).open("r") as f:
        conf  = yaml.safe_load(f)
    return conf

def get_sha_path_from_result(result_path_str : str):
    return result_path_str.replace(Path(result_path_str).stem, Path(result_path_str).stem + "_sha")

def get_resolved_path(path_str):
    return str(Path(path_str).resolve())