from pathlib import Path
from bisect import bisect
import pickle

import click
import json
import numpy as np
import pandas as pd

from common import util

def validate_file(ctx, param, value):
    path = Path(ctx.params['source_dir']) / (value + ".json")
    if not path.exists():
        raise click.BadParameter(f"No file present for model '{path.stem}', missing file: '{str(path.resolve())}'")
    
    return value + ".json"

def load_from_pkl(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data['vals'], data["normalized_vals"], data['cum_vals']

def save_pkl(path, vals, normalized_vals, cum_vals):
    data = {}
    data['vals'] = vals
    data['normalized_vals'] = normalized_vals
    data['cum_vals'] = cum_vals
    with path.open("wb") as f:
        pickle.dump(data, f)

def get_data(cache_dir, source_dir, model_name, metric):
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True, parents=True)

    metric_save_name = metric
    if metric == "prediction time":
        metric_save_name = "time"
    if metric == "time":
        metric = "prediction time"

    path = cache_path / (Path(model_name).stem + "_" + metric_save_name + ".pkl")
    if path.exists():
        vals, normalized_vals, cum_vals = load_from_pkl(path)
    else:
        src_path = Path(source_dir) / model_name
        with src_path.open("r") as f:
            runs = json.load(f)
        vals = sorted(run[metric] for run in runs)
        if metric == "prediction time":
            val_size = get_summed_val_size()
            vals = [val + val_size for val in vals] 
            
        normalized_vals = [val / max(vals) for val in vals]
        cum_vals = np.cumsum(normalized_vals)

        save_pkl(path, vals, normalized_vals, cum_vals) 

    return vals, normalized_vals, cum_vals



def sum_sizes_from_file(sha_path):
    config = util.get_config()
    
    val_size = 0
    with Path(sha_path).open() as f:
        shas = [line.strip() for line in f.readlines()]
    
    # Skip header
    shas = shas[1:]
    
    times = {}
    for times_csv in config['times_csvs']:
        curr_df = pd.read_csv(times_csv, header = None, names = ['sha', 'time'])
        for _, row in curr_df.iterrows():
            times[row['sha']] = row['time']
    
    for sha in shas:
        # Time is in microseconds, we work with seconds so converting
        val_size += times[sha] * 1e-6
        
    return val_size
    

def get_summed_val_size():
    cache_dir = util.get_config()["cdf_func_cache_dir"]
    cached_val_file = Path(cache_dir) / 'valsize.pkl'
    
    if cached_val_file.exists():
        with cached_val_file.open("rb") as f:
            val_size = pickle.load(f)
    else:
        config = util.get_config()        
        val_size = sum_sizes_from_file(util.get_sha_path_from_result(config['result_test_csv']))
        
        with cached_val_file.open("wb") as f:
            pickle.dump(val_size, f)
            
    return val_size
         
    

@click.command()
@click.option("--cache", "cache_dir", help = "Path to the cache dir, where the model's DF could be stored")
@click.option("--source", "source_dir", help = "Path to the directory containing the calculated source metrics (runtime, size, fmes, etc.)", type = click.Path(exists=True, file_okay=False, 
writable=True, readable=True))
@click.option("--model", "model_name", required = True, callback=validate_file, help = "The model to calculate the df for")
@click.option("--metric", help = "The metric to calculate the df for", required = True)
@click.option("--value", type = click.FLOAT)
def get(cache_dir, source_dir, model_name, metric, value):
    vals, normalized_vals, cum_vals = get_data(cache_dir, source_dir, model_name, metric)

    
    if metric == "time" or metric == "prediction time":
        value += get_summed_val_size()
    
    idx = bisect(vals, value) 
    if idx == 0:
        return 1
    
    normalized_res = cum_vals[idx - 1] / cum_vals[-1]
    
    orig_cum_vals = np.cumsum(vals)
    orig_res = orig_cum_vals[idx - 1] / orig_cum_vals[-1]
    
    #we need the inverted cdf so subtracting from 1
    return 1 - normalized_res


if __name__ == "__main__":
    get()