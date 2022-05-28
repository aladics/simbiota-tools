import portalocker
import json
import numpy as np
import click

from common import util
from evaluate.task_factory import TaskFactory
from evaluate import init_input

from dbh_api.res.runner import run_train_test_pair


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_as_done(task_id: str):
    already_done = []
    already_done_path = util.get_eval_progress_path()

    if already_done_path.exists():
        with portalocker.Lock(str(already_done_path), "r", timeout=60) as f:
            already_done = json.load(f)

    if task_id not in already_done:
        already_done.append(task_id)

    with portalocker.Lock(str(already_done_path), "w", timeout=60) as f:
        json.dump(already_done, f, indent=2)


def is_aready_done(task_id: str):
    eval_progress_path = util.get_eval_progress_path()
    if not eval_progress_path.exists():
        return False

    with portalocker.Lock(str(eval_progress_path), "r", timeout=60) as progress_json:
        already_done = json.load(progress_json)

    if task_id in already_done:
        return True

    return False


def get_best_hyper(usecase_type: str):
    with util.get_hyper_results_path().open() as f:
        hyper_results = json.load(f)

    return [result[usecase_type]["best_params"] for result in hyper_results.values()]


def run_on_all_learns(usecase_type: str, architecture: str, n_run: int, n_week: int):
    shared_params = {
        "label": "label",
        "resample": "none",
        "resample_amount": 0,
        "seed": 1337,
        "output": "output",
        "clean": False,
        "calc_completeness": True,
        "preprocess": [
            # [
            #    'features',
            #    'standardize'
            # ],
            ["labels", "binarize"]
        ],
        "return_results": True,
    }

    task_id = util.generate_task_id(architecture, n_run, n_week)
    if is_aready_done(task_id):
        return

    root_dir = util.get_eval_input_path() / architecture / str(n_run)
    train_csv = root_dir / f"{architecture}_merged_week_{n_week}_train.csv"
    test_csv = root_dir / f"{architecture}_merged_week_{n_week}_valid.csv"

    results = []
    factory = TaskFactory(shared_params)

    # Linear has to be done separately as it has not been hypertuned
    learn_task = factory.get("linear", task_id)
    run_train_test_pair(learn_task, train_csv, test_csv, results)

    for best_result in get_best_hyper(usecase_type):
        learn_task = factory.get(best_result, task_id)
        run_train_test_pair(learn_task, train_csv, test_csv, results)

    result_path = util.get_eval_results_path() / str(n_run) / f"{task_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w") as f:
        json.dump(results, f, cls=NpEncoder, indent=2)

    save_as_done(task_id)


def run_on_weeks(architecture: str, usecase_type: str):
    config = util.get_config()
    for n_run in range(config["start_run"], config["end_run"] + 1):
        for n_week in range(config["start_week"], config["end_week"] + 1):
            run_on_all_learns(usecase_type, architecture, n_run, n_week)


def run_on_architecture(architecture: str, usecase_type: str, init: bool):
    if init:
        init_input.init_all(architecture)
    run_on_weeks(architecture, usecase_type)


@click.command()
@click.option(
    "--usecase-type",
    help="The usecase type that correspond to the metaparams (alpha, beta etc.)",
    required=True,
)
@click.option(
    "--init/--no-init", default=False, help="Indicates if data should be initialized."
)
@click.option(
    "--single",
    help="Define a single run in the form of: <arch>-<n-run><n-week>",
    default=None,
)
def run(usecase_type, init, single):
    if single:
        pars = single.split("-")
        run_on_all_learns(usecase_type, pars[0], int(pars[1]), int(pars[2]))
    else:
        run_on_architecture("arm", usecase_type, init)
        run_on_architecture("mips", usecase_type, init)


if __name__ == "__main__":
    run()
