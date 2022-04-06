from redis import Redis
from rq import Queue
from evaluate import run as eval_run
from common import util

import click

@click.command()
@click.option("--usecase_type", help = "The usecase type that correspond to the metaparams (alpha, beta etc.)", required = True)
def run(usecase_type):
    queue = Queue(name = 'simbiota', connection = Redis())
    enqueue_all_architecutres_by_week(queue, usecase_type)

def enqueue_all_architecutres_by_week(queue, usecase_type):
    config = util.get_config()
    for arch in config['architecture_types']:
        for n_run in range(config['start_run'], config['end_run'] + 1):
            for n_week in range(config['start_week'], config['end_week'] + 1):
                queue.enqueue(eval_run.run_on_all_learns, args = (usecase_type, arch, n_run, n_week), job_timeout = "30m")            
                
if __name__ == "__main__":
    run()
    