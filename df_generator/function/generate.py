from asyncore import read
from email.policy import default
from pathlib import Path
from bisect import bisect
import pickle

import click
import json
import numpy as np

from numpy import require

def validate_file(ctx, param, value):
    path = Path(ctx.params['source_dir']) / (value + ".json")
    if not path.exists():
        raise click.BadParameter(f"No file present for model '{path.name}' in source dir '{path.parent}'")
    
    return value + ".json"

def load_from_pkl(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data['vals'], data['cum_vals']

def save_pkl(path, vals, cum_vals):
    data = {}
    data['vals'] = vals
    data['cum_vals'] = cum_vals
    with path.open("wb") as f:
        pickle.dump(data, f)

def get_data(cache_dir, source_dir, model_name, metric):
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True, parents=True)

    path = cache_path / (Path(model_name).stem + "_" + metric + ".pkl")
    if path.exists():
        vals, cum_vals = load_from_pkl(path)
    else:
        src_path = Path(source_dir) / model_name
        with src_path.open("r") as f:
            runs = json.load(f)
        vals = sorted(run[metric] for run in runs)
        cum_vals = np.cumsum(vals) 

        save_pkl(path, vals, cum_vals) 

    return vals, cum_vals

@click.command()
@click.option("--cache", "cache_dir", help = "Path to the result dir, where the model's DF will be stored")
@click.option("--source", "source_dir", help = "Path to the directory containing the calculated source metrics (runtime, size, fmes, etc.)", type = click.Path(exists=True, file_okay=False, 
writable=True, readable=True))
@click.option("--model", "model_name", required = True, callback=validate_file, help = "The model to calculate the df for")
@click.option("--metric", help = "The metric to calculate the df for", required = True)
@click.option("--value", type = click.INT)
def run(cache_dir, source_dir, model_name, metric, value):
    vals, cum_vals = get_data(cache_dir, source_dir, model_name, metric)

    idx = bisect(vals, value) 
    res = cum_vals[idx - 1] / cum_vals[-1]

    return res


if __name__ == "__main__":
    run()