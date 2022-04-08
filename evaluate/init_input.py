from pathlib import Path
from itertools import chain
from typing import List

import click

from common import util
from common.data import create_dataset

def get_file_part(weekly_prefix, n_run, n_week, suffix, label):
    prefix = weekly_prefix.replace("<label>", label)
    return Path(str(n_run)) / f"{prefix}{n_week}_{suffix}"

def get_weekly_files_to_merge(benign_root_path : Path, malware_root_path : Path, n_run : int, n_weeks : List[int], architecture : str, suffix : str, config):
    weekly_prefix = config['weekly_prefix'].replace("<architecture>", architecture)
    files_to_merge = []

    for n_week in n_weeks:
        bening_file_path = get_file_part(weekly_prefix, n_run, n_week, suffix, "b")
        malware_file_path = get_file_part(weekly_prefix, n_run, n_week, suffix, "mw")
        files_to_merge += [benign_root_path / bening_file_path, malware_root_path / malware_file_path]

    return files_to_merge

def init_result_path(architecture, n_run, n_week, suffix):
    config = util.get_config()
    weekly_prefix = config['weekly_prefix'].replace("<architecture>", architecture)
    result_path = util.get_eval_input_path() / architecture / get_file_part(weekly_prefix, n_run, n_week, suffix, "merged")
    result_path.parent.mkdir(parents = True, exist_ok = True)

    return result_path


def init_train_files(architecture : str):
    config = util.get_config()
    arch_root_path = Path(config['weekly_root_dir']) / architecture

    benign_root_path = arch_root_path / "benign" / "weekly"
    malware_root_path = arch_root_path / "malware" / "weekly"

    bening_filename = config['boosting_filename'].replace("<architecture>", architecture).replace("<label>", "b")
    benign_boosting_path = Path(config['weekly_root_dir']) / architecture /  "benign" / f"{bening_filename}.csv"
    malware_filename = config['boosting_filename'].replace("<architecture>", architecture).replace("<label>", "mw")
    malware_boosting_path = Path(config['weekly_root_dir']) / architecture /  "malware" / f"{malware_filename}.csv"

    for n_run in range(config['start_run'], config['end_run'] + 1):
        for n_week in range(config['start_week'], config['end_week'] +1):
            files_to_merge = [benign_boosting_path, malware_boosting_path]

            weeks_to_merge = [*range(config['start_week'], n_week)]
            if weeks_to_merge:
                files_to_merge += get_weekly_files_to_merge(benign_root_path, malware_root_path, n_run, weeks_to_merge, architecture, "train.csv", config)

            files_to_merge = [str(file_path.resolve()) for file_path in files_to_merge]

            result_path = init_result_path(architecture, n_run, n_week, "train.csv")
            create_dataset.main([*files_to_merge, str(result_path.resolve())], standalone_mode = False)


def init_valid_files(architecture : str):
    config = util.get_config()
    arch_root_path = Path(config['weekly_root_dir']) / architecture

    benign_root_path = arch_root_path / "benign" / "weekly"
    malware_root_path = arch_root_path / "malware" / "weekly"

    for n_run in range(config['start_run'], config['end_run'] + 1):
        for n_week in range(config['start_week'], config['end_week'] +1):
            files_to_merge = get_weekly_files_to_merge(benign_root_path, malware_root_path, n_run, [n_week], architecture, 'valid.csv', config)

            result_path = init_result_path(architecture, n_run, n_week, "valid.csv")
            files_to_merge = [str(file_path.resolve()) for file_path in files_to_merge]
            create_dataset.main([*files_to_merge, str(result_path.resolve())], standalone_mode = False)



@click.command()
@click.option('--architecture', required = True)
def init_all(architecture : str):
    init_valid_files(architecture)
    init_train_files(architecture)

if __name__ == "__main__":
    init_all()
