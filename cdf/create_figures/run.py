from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import click
import numpy as np
import json

import common.util as util

@click.command()
@click.option("--bins", "bins_", default=10, type=click.INT)
@click.option("--res-dir", default = "cdf/create_figures/results")
def run(bins_, res_dir):
    config = util.get_config()
    res_path = Path(res_dir)
    res_path.mkdir(exist_ok=True)
    
    for pkl_path in Path(config['cdf_func_cache_dir']).glob("*.pkl"):
        model_name = pkl_path.stem
        
        if not (model_name.endswith("memory") or model_name.endswith("time")):
            print(f"Skipping unparsable file '{str(pkl_path)}'")
            continue
        
        model_path = res_path / model_name.split("_")[0]
        model_path.mkdir(exist_ok=True)
        with pkl_path.open("rb") as f:
            cached_vals = pickle.load(f)['vals']
            
            fig, axs = plt.subplots(2, sharex=True)
            fig.suptitle(model_name)
            
            count, bins_count = np.histogram(cached_vals, bins=bins_)
            pdf = count / sum(count)
            pdf = np.insert(pdf, 0, 0)
            cdf = np.cumsum(pdf)
            
            axs[0].hist(cached_vals, bins=bins_,  alpha=0.7, facecolor='blue', ec = "darkblue")
            axs[0].set_title('Histogram')
            
            axs[1].plot(bins_count, pdf, color="red", label="PDF")
            axs[1].plot(bins_count, cdf, color="green", label="CDF")
            
            if model_name.endswith("memory"):
                axs[1].set(xlabel = "Bytes")
            elif model_name.endswith("time"):
                axs[1].set(xlabel = "Seconds")
            
            with (model_path / f"{model_name}.json").open("w") as f:
                json.dump(cached_vals, f)
            
            savefig_path_str = str((model_path / f"{model_name}.pdf").resolve())
            
            fig.legend()
            fig.savefig(savefig_path_str)
            
            plt.close(fig)
            
            
if __name__ == "__main__":
    run()