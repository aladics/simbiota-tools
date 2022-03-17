from pathlib import Path

import cdf.function.function as cdf_function
import common.util as util

if __name__ == "__main__":
    config = util.get_config()
    
    cache_dir = util.get_resolved_path(config['cdf_func_cache_dir'])
    source_dir = util.get_resolved_path(config['cdf_results_dir'])
    
    model_name = "forest"
    metric_name = "time"
    value = 9.33
    
    cdf_func_params = ["--cache", util.get_resolved_path(config['cdf_func_cache_dir']), "--source", util.get_resolved_path(config['cdf_results_dir']),
                       "--model", "forest", "--metric", "time", "--value", 9.33]
    
    res = cdf_function.get(cdf_func_params, standalone_mode = False)
    print(f"From test: {res}")
