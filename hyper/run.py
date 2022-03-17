from pathlib import Path

import yaml
import json

import common.util as util
import common.model_generator as model_generator
from cdf.function.function import get as get_cdf_value

def generate_models_for_hyper(config):
    search_params_path = Path(config['search_params_file'])
    train_path = Path(config['result_train_csv'])
    test_path = Path(config['result_test_csv'])
    results_path = Path(config['hyper_results_dir'])
    
    results_path.mkdir(exist_ok=True, parents=True)
    
    return model_generator.run(["--search-params", str(search_params_path.resolve()), "--train-file", str(train_path.resolve()), "--test-file",
    str(test_path.resolve()), "--n", config['n_models_for_hyper'], "--save-path", str(results_path.resolve())], standalone_mode = False)

def get_score(metaparam_vals, parsed_result, model_name, config):
    prec = metaparam_vals['alpha'] * parsed_result['test']['precision']
    recall = metaparam_vals['beta'] * parsed_result['test']['recall']
    
    cdf_func_params = ["--cache", util.get_resolved_path(config['cdf_func_cache_dir']), "--source", util.get_resolved_path(config['cdf_results_dir']),
                       "--model", model_name, "--metric", "memory", "--value", parsed_result['memory']]
    memory = metaparam_vals['gamma'] * get_cdf_value(cdf_func_params, standalone_mode = False)
    
    cdf_func_params[7] = "time"
    cdf_func_params[9] = parsed_result['prediction time']
    
    time = metaparam_vals['delta'] * get_cdf_value(cdf_func_params, standalone_mode = False)
    
    return parsed_result['strategy'], prec + recall + memory + time
    
def parse_model_result(path_str: Path):
    with path_str.open() as f:
        return yaml.safe_load(f)
    

def generate_hyper_scores(config):
    scores = {}
    for hyper_res in Path(config['hyper_results_dir']).glob("*.json"):
        if hyper_res.stem == "results":
            continue
        model_name = hyper_res.stem
        parsed_res = parse_model_result(hyper_res)
        scores[model_name] = {}
        
        for metaparam_name in config['hyper_metaparams']:
                scores[model_name][metaparam_name] = {}
                scores[model_name][metaparam_name]['all'] = {}
                scores[model_name][metaparam_name]['best_score'] = 0
    
        for model_data in parsed_res:
            for metaparam_name, metaparam_vals in config['hyper_metaparams'].items():
                model_params, score = get_score(metaparam_vals, model_data, model_name, config)
                scores[model_name][metaparam_name]['all'][model_params] = score

                if not scores[model_name][metaparam_name]['best_score'] or scores[model_name][metaparam_name]['best_score'] > score :
                    scores[model_name][metaparam_name]['best_score'] = score
                    scores[model_name][metaparam_name]['best_params'] = model_params
    
    with (Path(config['hyper_results_dir']) / 'results.json').open("w") as f:
        json.dump(scores, f)        

def main():
    config = util.get_config()
    # generate_models_for_hyper(config)
    generate_hyper_scores(config)

if __name__ == "__main__":
    main()