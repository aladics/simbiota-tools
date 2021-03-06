from pathlib import Path

import pandas as pd
import click

import common.util as util

RANDOM_SEED = 1234


def generate_header():
    header = ["sha"]
    for i in range(0, util.get_feature_num()):
        header.append(f"feature_{i}")
    header.append("label")

    return header


@click.command()
@click.argument(
    "sources", nargs=-1, type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.argument("result")
def main(sources, result: str):
    """Merge and prepare feature csv files (SOURCES) and save them in a file (RESULT)."""
    dfs = (pd.read_csv(src, header=None, delimiter=",") for src in sources)

    # Merge input CSVs, permutate rows, add header
    final_df = pd.concat(dfs).sample(frac=1, random_state=RANDOM_SEED)
    final_df.columns = generate_header()

    sha_df = final_df["sha"]

    # Drop SHA row
    final_df = final_df.drop(labels="sha", axis=1)

    # Drop rows with NaN values
    final_df = final_df.dropna()

    # Flip labels
    final_df["label"] = (~(final_df["label"].astype(bool))).astype(int)

    sha_df.to_csv(Path(util.get_sha_path_from_result(result)).resolve(), index=False)
    final_df.to_csv(Path(result).resolve(), index=False)


if __name__ == "__main__":
    main()
