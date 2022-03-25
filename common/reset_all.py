from email.policy import default
from common import util
from pathlib import Path
import shutil

import click

def recreate_empty_dir(dir_path: Path):
    """ Deletes dir_path if exists then recreates it as empty. """
    
    if dir_path.exists():
        shutil.rmtree(str(dir_path.resolve()))
    dir_path.mkdir(exist_ok=True, parents=True)

@click.command()
@click.option("--mode", type=click.Choice(['all', 'hyper', 'data', 'cdf'], case_sensitive = False), default = 'all')
def run(mode):
    config = util.get_config()
    
    if mode == "all" or mode == "data":
        Path(config['result_train_csv']).unlink()
        Path(config['result_test_csv']).unlink()
    
    if mode == "all" or mode == "cdf":
        recreate_empty_dir(Path(config['cdf_results_dir']))
        recreate_empty_dir(Path(config['cdf_func_cache_dir']))
        
    if mode == "all" or mode == "hyper":
        recreate_empty_dir(Path(config['hyper_results_dir']))
    
    
if __name__ == "__main__":
    run()
