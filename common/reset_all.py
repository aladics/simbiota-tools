from common import util
from pathlib import Path
import shutil

import click


def recreate_empty_dir(dir_path: Path):
    """Deletes dir_path if exists then recreates it as empty."""

    if dir_path.exists():
        shutil.rmtree(str(dir_path.resolve()))
    dir_path.mkdir(exist_ok=True, parents=True)


@click.command()
@click.option(
    "--mode",
    type=click.Choice(
        ["all", "hyper", "hyper_data", "cdf", "eval"], case_sensitive=False
    ),
    default="all",
)
def run(mode):
    config = util.get_config()

    if mode == "all" or mode == "hyper_data":
        Path(config["result_train_csv"]).unlink()
        Path(config["result_test_csv"]).unlink()

    if mode == "all" or mode == "cdf":
        recreate_empty_dir(Path(config["cdf_results_dir"]))
        recreate_empty_dir(Path(config["cdf_func_cache_dir"]))

    if mode == "all" or mode == "hyper":
        recreate_empty_dir(Path(config["hyper_results_dir"]))

    if mode == "all" or mode == "eval":
        recreate_empty_dir(util.get_eval_input_path())
        recreate_empty_dir(util.get_eval_results_path())
        recreate_empty_dir(Path(config["sandbox_root"]))
        util.get_eval_progress_path().unlink()

    print("Environment reset done.")


if __name__ == "__main__":
    run()
