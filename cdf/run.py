from pathlib import Path

from common.data import create_dataset
from common import model_generator
from common.util import get_config, is_standardize


############################################# CONSTANS #############################################
CONFIG_PATH = "common/config.yaml"

RAW_TRAIN_CSVS = None
TRAIN_CSV = None
RAW_TEST_CSVS = None
TEST_CSV = None
SEARCH_PARAMS = None
RESULTS = None

#####################################################################################################


def load_constants(conf):
    global RAW_TRAIN_CSVS, TRAIN_CSV, RAW_TEST_CSVS, TEST_CSV, SEARCH_PARAMS, RESULTS

    RAW_TRAIN_CSVS = conf["raw_train_csvs"]
    TRAIN_CSV = conf["result_train_csv"]
    RAW_TEST_CSVS = conf["raw_test_csvs"]
    TEST_CSV = conf["result_test_csv"]
    SEARCH_PARAMS = conf["search_params_file"]
    RESULTS = conf["cdf_results_dir"]


def generate_models_for_cdf(train_path, test_path, results_path, search_params, n):
    standardize_str = "--dont-standardize"
    if is_standardize():
        standardize_str = "--standardize"

    return model_generator.run(
        [
            "--search-params",
            search_params,
            "--train-file",
            str(train_path.resolve()),
            "--test-file",
            str(test_path.resolve()),
            "--n",
            n,
            "--save-path",
            str(results_path.resolve()),
            standardize_str,
        ],
        standalone_mode=False,
    )


def init_dirs(train_path: Path, test_path: Path, results_path: Path):
    train_path.parent.mkdir(exist_ok=True, parents=True)
    test_path.parent.mkdir(exist_ok=True, parents=True)
    results_path.mkdir(exist_ok=True, parents=True)


def main():
    conf = get_config()
    load_constants(conf)

    train_path = Path(TRAIN_CSV)
    test_path = Path(TEST_CSV)
    results_path = Path(RESULTS)
    search_params_path = Path(SEARCH_PARAMS)

    init_dirs(train_path, test_path, results_path)

    if not train_path.exists():
        create_dataset.main([*RAW_TRAIN_CSVS, TRAIN_CSV], standalone_mode=False)

    if not test_path.exists():
        create_dataset.main([*RAW_TEST_CSVS, TEST_CSV], standalone_mode=False)

    times = {}
    times[conf["n_models_for_cdf"]] = generate_models_for_cdf(
        train_path,
        test_path,
        results_path,
        search_params_path,
        conf["n_models_for_cdf"],
    )

    print(times)


if __name__ == "__main__":
    main()
