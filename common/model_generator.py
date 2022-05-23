import sys
from pathlib import Path
import itertools
from shutil import rmtree
import random
from copy import deepcopy
import time
import logging
from math import ceil

import yaml
import click
import numpy as np

from common import util
 
util.set_dbh_path()
from dbh_api.res import params
from dbh_api.res.runner import LearnTask, run_train_test_pair, save_results_to_json


class TaskFactory:
    class DNNCLearnTask(LearnTask):
        def __init__(self, shared_params, sandbox):
            super().__init__(shared_params)
            self.sandbox = sandbox


        def pre_run(self):
            Path(self.sandbox).mkdir(parents=True, exist_ok=True)

        def post_run(self):
            if Path(self.sandbox).exists():
                rmtree(str(Path(self.sandbox).resolve()))

    def __init__(self, shared_params):
        self.shared_params = shared_params

    def get(self, model_name, sargs):
        if model_name != "cdnnc" and model_name != "sdnnc":
            learn_task = LearnTask(self.shared_params)
        else:
            sandbox_path = Path("sandboxes")
            sandbox_path.mkdir(exist_ok=True, parents=True)
            sargs['sandbox'] = str(sandbox_path.resolve())

            learn_task = self.DNNCLearnTask(self.shared_params, str((sandbox_path / model_name).resolve()))
       
        model_params = None
        if model_name == "svm":
            model_params =  params.SVMParams(**sargs)            
        if model_name == "forest":
            model_params =  params.ForestParams(**sargs)
        if model_name == "sdnnc":
            model_params = params.SDNNCParams(**sargs)
        if model_name == "cdnnc":
            model_params = params.CDNNCParams(**sargs)
        if model_name == "adaboost":
            model_params = params.AdaboostParams(**sargs)
        if model_name == "knn":
            model_params = params.KNNParams(**sargs)
        if model_name == "logistic":
            model_params = params.LogisticParams(**sargs)
        if model_name == "linear":
            model_params = params.LinearParams(**sargs)
        if model_name == "tree":
            model_params = params.TreeParams(**sargs)
                    
            
        learn_task.params = model_params.get()
        learn_task.set_save_model_path("")

        return learn_task


def dict_product(**dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def generate_random_nums_with_sum(n, num_sum):
    res = [-1]
    while sum(res) != num_sum:
        nums = np.random.random_sample(n)
        logits = [num / sum(nums) for num in nums]
        res = [round(num * num_sum) for num in logits]

    return res


def generate_all_param_configs(parsed_params_list, n):
    portions = generate_random_nums_with_sum(len(parsed_params_list), n)
    all_configs = []
    for i in range(len(parsed_params_list)):
        config = generate_param_configs(parsed_params_list[i], portions[i])
        all_configs.append(config)

    return all_configs


def generate_param_configs(parsed_params, n):
    generated_params = {}
    for par_name, par_config in parsed_params.items():
        if par_config['type'] == 'uniform_int':
            generated_params[par_name] = np.random.random_integers(int(par_config["from"]), int(par_config['to']), n).tolist()
        if par_config['type'] == 'uniform_float':
            generated_params[par_name] = np.random.uniform(float(par_config["from"]), float(par_config['to']), n).tolist()
        if par_config['type'] == 'multi':
            generated_params[par_name] = random.choices(par_config['values'], k = n)
    
    return generated_params


def generate_configs(param_configs):
    configs = []
    for param_config in param_configs:
        configs += [dict(zip(param_config.keys(), param)) for param in zip(*param_config.values())]
    
    return configs

def run_task(learn_task, results, train_file, test_file):
    train_path = Path(train_file)
    test_path = Path(test_file)
    run_train_test_pair(learn_task, train_path, test_path, results)
    return

def parse_params(raw_params):
    param_configs = [{}]
    
    independent_pars = {}
    dependent_pars = {}
    for par_name, par_config in raw_params.items():
        if "depends_on" in par_config:
            dependent_pars[par_name] = par_config
        else:
            independent_pars[par_name] = par_config

    for key, val in independent_pars.items():
        param_configs[0][key] = val.copy()

    for key, val in dependent_pars.items():
        config = deepcopy(param_configs[0])
        config[key] = val
        for dependency_name, dependency_details in val['depends_on'].items():
            config[dependency_name]['values'] = [dependency_details['values']]
            param_configs[0][dependency_name]['values'].remove(dependency_details['values'])
        param_configs.append(config)
    
    return param_configs


def generate_model(model_name, raw_params, shared_params, save_path, train_file, test_file, n):
    factory = TaskFactory(shared_params)

    parsed_params = parse_params(raw_params)
    param_configs = generate_all_param_configs(parsed_params, n)
    configs = generate_configs(param_configs)

    results = []
    times = []
    for config in configs:
        
        learn_task = factory.get(model_name, config)
        start = time.time()
        run_task(learn_task, results, train_file, test_file)
        times.append(round(time.time() - start, 2))
    
    save_root = Path(save_path)
    if not save_root.exists():
        save_root.mkdir(exist_ok=True, parents=True)

    save_results_to_json(results, save_root / (f"{model_name}.json"))
    return times
    
def get_logger(log_file):
    logger = logging.getLogger('df_generator')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s   %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

@click.command()
@click.option("--search-params", default="common/search_params.yaml")
@click.option("--save-path", default = "save_files")
@click.option("--train-file", required=True)
@click.option("--test-file", required=True)
@click.option("--n", default = 100, help = "Number of randomly parametrized models to generate")
@click.option("--log-file", default = "common/model_gen_progress.log")
def run(search_params, save_path, train_file, test_file, n, log_file):
    logger = get_logger(log_file)
    
    with Path(search_params).open() as f:
        params = yaml.safe_load(f)

    
    shared_params = {
        'label' : 'label',
        'resample' : 'none',
        'resample_amount' : 0,
        'seed' : 1337,
        'output' : 'output',
        'clean' : False,
        'calc_completeness' : True,
        'preprocess': [
            #[
            #    'features',
            #    'standardize'
            #],
            [
                'labels',
                'binarize'
            ]
        ],
        'return_results' : True        
    }

    time_stats = {}
    for key, val in params.items():
        start = time.time()
        times = generate_model(key, val, shared_params, save_path, train_file, test_file, n)
        time_stats[key] = time.time() - start
        
        logger.info(f" {key}(rand_n = {n}) done. Elapsed: {round(time_stats[key], 3)} secs ({ceil(time_stats[key] / 60)} mins).\nExecution times per model:\n{str(times)}" )

    return time_stats
    

if __name__ == "__main__":
    run()
    
    pass