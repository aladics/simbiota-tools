from genericpath import exists
from pathlib import Path

from files import create_dataset
from data import df_generator
import yaml


############################################# CONSTANS #############################################
CONFIG_PATH     = "config.yaml"

RAW_TRAIN_CSVS  = None
TRAIN_CSV       = None
RAW_TEST_CSVS   = None
TEST_CSV        = None
SEARCH_PARAMS   = None
RESULTS         = None

#####################################################################################################

def load_config():
    global RAW_TRAIN_CSVS, TRAIN_CSV, RAW_TEST_CSVS, TEST_CSV, SEARCH_PARAMS, RESULTS
    with Path(CONFIG_PATH).open("r") as f:
        conf  = yaml.safe_load(f)
    
    RAW_TRAIN_CSVS = conf['raw_train_csvs']
    TRAIN_CSV = conf['result_train_csv']
    RAW_TEST_CSVS = conf['raw_test_csvs']
    TEST_CSV = conf['result_test_csv']
    SEARCH_PARAMS = conf['search_params_file']
    RESULTS = conf['json_results_dir']
    

def run_df_generator(train_path, test_path, results_path, search_params, n):
    return df_generator.run(["--search-params", search_params, "--train-file", str(train_path.resolve()), "--test-file",
    str(test_path.resolve()), "--n", n, "--save-path", str(results_path.resolve())], standalone_mode = False)

def init_dirs(train_path : Path, test_path : Path, results_path : Path):
    train_path.parent.mkdir(exist_ok=True, parents=True)
    test_path.parent.mkdir(exist_ok=True, parents=True)
    results_path.mkdir(exist_ok=True, parents=True)


def main():
    load_config()
    
    train_path = Path(TRAIN_CSV)
    test_path = Path(TEST_CSV)
    results_path = Path(RESULTS)
    search_params_path = Path(SEARCH_PARAMS)

    init_dirs(train_path, test_path, results_path)

    if not train_path.exists():
        create_dataset.main([*RAW_TEST_CSVS, TRAIN_CSV], standalone_mode = False)
    
    if not test_path.exists():
        create_dataset.main([*RAW_TEST_CSVS, TEST_CSV], standalone_mode = False)

    times = {}
    times["10"] = run_df_generator(train_path, test_path, results_path, search_params_path, 10)
    #times["100"] = run_df_generator(train_path, test_path, results_path, search_params_path, 100)
    #times["1000"] = run_df_generator(train_path, test_path, results_path, search_params_path, 1000)

    print(times)
    
    

if __name__ == "__main__":
    main()