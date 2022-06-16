from pathlib import Path
import click

from common.data import create_dataset
from common import model_generator
from common.util import get_config


############################################# CONSTANS #############################################
CONFIG_PATH = "common/config.yaml"

RAW_TRAIN_CSVS = []
TRAIN_CSV = ""
RAW_TEST_CSVS = []
TEST_CSV = ""
SEARCH_PARAMS = ""
RESULTS = ""

#####################################################################################################


def load_constants(conf) -> None:
    global RAW_TRAIN_CSVS, TRAIN_CSV, RAW_TEST_CSVS, TEST_CSV, SEARCH_PARAMS, RESULTS

    RAW_TRAIN_CSVS = conf["raw_train_csvs"]
    TRAIN_CSV = conf["result_train_csv"]
    RAW_TEST_CSVS = conf["raw_test_csvs"]
    TEST_CSV = conf["result_test_csv"]
    SEARCH_PARAMS = conf["search_params_file"]
    RESULTS = conf["cdf_results_dir"]


def generate_models_for_cdf(
    train_path: Path, test_path: Path, results_path: Path, search_params : Path, n : int
) -> None:
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
        ],
        standalone_mode=False,
    )


def create_train_test_sets() -> None:
    create_dataset.main([*RAW_TRAIN_CSVS, TRAIN_CSV], standalone_mode=False)
    create_dataset.main([*RAW_TEST_CSVS, TEST_CSV], standalone_mode=False)


def init_dirs(train_path: Path, test_path: Path, results_path: Path) -> None:
    train_path.parent.mkdir(exist_ok=True, parents=True)
    test_path.parent.mkdir(exist_ok=True, parents=True)
    results_path.mkdir(exist_ok=True, parents=True)


@click.command()
@click.option(
    "--only-datasets/--not-only-datasets",
    help="Create only datasets or run the cdf generation too",
    default=False,
)
def main(only_datasets : bool):
    conf = get_config()
    load_constants(conf)

    train_path = Path(TRAIN_CSV)
    test_path = Path(TEST_CSV)
    results_path = Path(RESULTS)
    search_params_path = Path(SEARCH_PARAMS)

    init_dirs(train_path, test_path, results_path)
    create_train_test_sets()

    if not only_datasets:
        times = {}
        times[conf["n_models_for_cdf"]] = generate_models_for_cdf(
            train_path,
            test_path,
            results_path,
            search_params_path,
            conf["n_models_for_cdf"],
        )


if __name__ == "__main__":
    main()
