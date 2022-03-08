from genericpath import exists
from pathlib import Path
import time

from files import create_dataset
from data import df_generator

########################################## DATASET RELATED ##########################################
RAW_TRAIN_CSVS = ["F:/work/kutatas/simbiota/df_generator/files/tlsh/tlsh/arm/benign/hyper/arm_b_hyper_train.csv", 
                  "F:/work/kutatas/simbiota/df_generator/files/tlsh/tlsh/arm/malware/hyper/arm_mw_hyper_train.csv"]
TRAIN_CSV = "data/final/train.csv"
RAW_TEST_CSVS = ["F:/work/kutatas/simbiota/df_generator/files/tlsh/tlsh/arm/benign/hyper/arm_b_hyper_valid.csv", 
                  "F:/work/kutatas/simbiota/df_generator/files/tlsh/tlsh/arm/malware/hyper/arm_mw_hyper_valid.csv"]
TEST_CSV = "data/final/test.csv"

########################################### MODEL RELATED ###########################################
SEARCH_PARAMS = "data/search_params.yaml"
RESULTS = "data/final/results"

#####################################################################################################

def run_df_generator(train_path, test_path, results_path, search_params, n):
    return df_generator.run(["--search-params", search_params, "--train-file", train_path.resolve(), "--test-file",
    test_path.resolve(), "--n", n, "--save-path", results_path.resolve()], standalone_mode = False)

def init_dirs(train_path : Path, test_path : Path, results_path : Path):
    train_path.parent.mkdir(exist_ok=True, parents=True)
    test_path.parent.mkdir(exist_ok=True, parents=True)
    results_path.mkdir(exist_ok=True, parents=True)


def main():
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