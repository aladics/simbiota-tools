"""Runs and stores results for all use-cases/hyper metaparams"""

import subprocess
import time
import logging
from typing import List, Union

import click


from common import util


#### SHELL COMMDANDS ###

RESET_EVAL_DIRS = f"""
. /home/aladics/miniconda3/etc/profile.d/conda.sh
conda activate dwf_client
python -m common.reset --mode='eval-progress'
"""

ENQUEUE_USECASE = f"""
. /home/aladics/miniconda3/etc/profile.d/conda.sh
conda activate dwf_client
python -m evaluate.rq_enqueue --usecase-type='%s'
"""

GENERATE_RANKINGS = f"""
. /home/aladics/miniconda3/etc/profile.d/conda.sh
conda activate dwf_client
python -m evaluate.generate_rankings --usecase-type='%s'
"""

STORE_RESULTS = f"""
. /home/aladics/miniconda3/etc/profile.d/conda.sh
conda activate dwf_client
mkdir -p %s
cp -r {util.get_eval_results_path()} %s
cp -r {util.get_eval_ranks_path()} %s
"""

GET_QUE_LEFT = r"""
. /home/aladics/miniconda3/etc/profile.d/conda.sh
conda activate dwf_client
rq info -R | head -n 1 | awk {'print $3'}
"""

### OTHER CONSTANTS ###
TIMEOUT = 60 * 5


logger = logging.getLogger('run_all_usecases')
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s | %(asctime)s] %(message)s",
    datefmt='%Y/%d/%m %H:%M'
)

def convert_to_dir_name(usecase_name : str) -> str:
    """
    Gets the directory name corresponding to a usecase.
    """
    return usecase_name.replace(" ", "_").lower()

def run_sh_command(command : str) -> subprocess.CompletedProcess:
    """
    Run a shell command, return the results.
    """
    process = subprocess.run(command, stderr = subprocess.PIPE, stdout = subprocess.PIPE,  shell = True)
    return process


def get_store_results_query(usecase_name: str) -> str:
    """
    Get the store results shell command with the deliverable dir interpolated.
    """
    deliverable_eval_dir = util.get_deliverable_dir() / "eval" / convert_to_dir_name(usecase_name)

    return STORE_RESULTS % (deliverable_eval_dir, deliverable_eval_dir, deliverable_eval_dir)

def get_finished_usecases(usecases_str : str) -> List[str]:
    if not usecases_str:
        return []
    else:
        return [usecase.strip() for usecase in usecases_str.split(",")]

def get_remaining_usecases(finished_usecases : List[str]) -> List[str]:
    return [usecase for usecase in util.get_usecase_names() if usecase not in finished_usecases]

def get_next_usecase(finished_usecases : List[str]) -> Union[str,  None]:
    usecase_names = util.get_usecase_names()
    return next((usecase for usecase in usecase_names if usecase not in finished_usecases), None)

def get_remaining_jobs() -> int:
    return int(run_sh_command(GET_QUE_LEFT).stdout.strip())

def monitor_usecase_status() -> None:
    """
    Blocks the program until the current use-case is done (that is, the simbiota rq queue is empty).
    """
    while True:
        remaining_jobs = get_remaining_jobs()
        if remaining_jobs == 0:
            logger.info("Current usecase is evaluated.")
            break
        else:
            logger.info(f"Remaining job count: {remaining_jobs}. Waiting for {TIMEOUT} seconds.")
            time.sleep(TIMEOUT)

def finalize(usecase : str) -> None:
    """
    Finalize a usecase by generating the rankings and saving the results and rankings to a 'deliverables' directory.
    """

    logger.info(f"Finalizing usecase '{usecase}'...")
    res = run_sh_command(GENERATE_RANKINGS % usecase)
    logger.info(f"Generate rankings done\n --- stderr: {res.stderr.decode('utf-8').strip()}\n --- stdout: {res.stdout.decode('utf-8').strip()}")
    res = run_sh_command(get_store_results_query(usecase))
    logging.info(f"Store results done\n --- stderr: {res.stderr.decode('utf-8').strip()}\n --- stdout: {res.stdout.decode('utf-8').strip()}")

def eval_usecase(usecase : str, finished_usecases : List[str]) -> None:
    """
    Wait for a usecase to finish evaluating, then finalize it.
    """

    logger.info(f"Started evaluating usecase '{usecase}")
    monitor_usecase_status()
    finalize(usecase)
    finished_usecases.append(usecase)
    logger.info(f"Finalizing usecase '{usecase}' is done.")


def reset_eval_progress() -> None:
    res = run_sh_command(RESET_EVAL_DIRS)
    logger.info(f"Resetting eval directories is done.\n --- stderr: {res.stderr.decode('utf-8').strip()}\n --- stdout: {res.stdout.decode('utf-8').strip()} ")

def enqueue_jobs(usecase : str) -> None:
    """
    Enqueue all the jobs related to a usecase.
    """

    res = run_sh_command(ENQUEUE_USECASE % usecase)
    logger.info(f"Enqueuing jobs for usecase '{usecase}' done.\n --- stderr: {res.stderr.decode('utf-8').strip()}\n --- stdout: {res.stdout.decode('utf-8').strip()}")

@click.command()
@click.option("--finished-usecases", "finished_usecases_str", default = "", help="Provide already finished usecases as a comma separated list.")
@click.option("--current-usecase", default = None, help = "The usecase that is currently being evaluated.")
def main(finished_usecases_str :str , current_usecase : str):
    logger.info(f"Starting work on feature set '{util.get_feature_set_name()}'")
    finished_usecases = get_finished_usecases(finished_usecases_str)
    logger.info(f"Finished usecases: {finished_usecases}")
    logger.info(f"Remaining usecasees: {get_remaining_usecases(finished_usecases)}")

    if current_usecase:
        eval_usecase(current_usecase, finished_usecases)

    while True:
        current_usecase = get_next_usecase(finished_usecases)
        if not current_usecase:
            break

        logger.info(f"Starting work on usecase '{current_usecase}'")
        reset_eval_progress()
        enqueue_jobs(current_usecase)
        eval_usecase(current_usecase, finished_usecases)

    logger.info("Finished working on feature set '{util.get_feature_set_name()}'")


if __name__ == "__main__":
    main()
