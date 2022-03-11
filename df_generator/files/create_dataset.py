from pathlib import Path

import pandas as pd
import click

RANDOM_SEED = 1234

def generate_header():
    header = ["sha"]
    for i in range(0,127):
        header.append(f"feature_{i}")
    header.append("label")

    return header


@click.command()
@click.argument('sources', nargs =-1, type = click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('result')
def main(sources, result):
    """Merge and prepare tlsh csv files (SOURCES) and save them in a file (RESULT)."""
    dfs = (pd.read_csv(src, header=None, delimiter=",") for src in sources)
    
    # Merge input CSVs, permutate rows, add header
    final_df = pd.concat(dfs).sample(frac=1, random_state=RANDOM_SEED)
    final_df.columns = generate_header()
    
    # Drop SHA row
    final_df = final_df.drop(labels = "sha", axis = 1)
    
    # Flip labels
    final_df['label'] = (~(final_df['label'].astype(bool))).astype(int)
    
    final_df.to_csv(Path(result).resolve(), index = False)
    
if __name__ == "__main__":
    main()