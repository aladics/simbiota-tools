from pathlib import Path

import yaml
import json
import click

import common.util as util
import common.model_generator as model_generator


def generate_models_for_hyper(config):
    search_params_path = Path(config["search_params_file"])
    train_path = Path(config["result_train_csv"])
    test_path = Path(config["result_test_csv"])
    results_path = Path(config["hyper_results_dir"])

    results_path.mkdir(exist_ok=True, parents=True)

    return model_generator.run(
        [
            "--search-params",
            str(search_params_path.resolve()),
            "--train-file",
            str(train_path.resolve()),
            "--test-file",
            str(test_path.resolve()),
            "--n",
            config["n_models_for_hyper"],
            "--save-path",
            str(results_path.resolve()),
        ],
        standalone_mode=False,
    )


def parse_model_result(path_str: Path):
    with path_str.open() as f:
        return yaml.safe_load(f)


def generate_hyper_scores(config):
    scores = {}
    for hyper_res in Path(config["hyper_results_dir"]).glob("*.json"):
        if hyper_res.stem == "results":
            continue
        model_name = hyper_res.stem
        parsed_res = parse_model_result(hyper_res)
        scores[model_name] = {}

        for usecase_name in config["usecases"]:
            scores[model_name][usecase_name] = {}
            scores[model_name][usecase_name]["all"] = {}
            scores[model_name][usecase_name]["best_score"] = 0

        for model_data in parsed_res:
            for usecase_name, metaparam_vals in config["usecases"].items():
                model_params, score = util.get_u_score(
                    metaparam_vals, model_data, model_name
                )
                scores[model_name][usecase_name]["all"][model_params] = score

                if scores[model_name][usecase_name]["best_score"] < score:
                    scores[model_name][usecase_name]["best_score"] = score
                    scores[model_name][usecase_name]["best_params"] = model_params

    with (Path(config["hyper_results_dir"]) / "results.json").open("w") as f:
        json.dump(scores, f)


@click.command()
@click.option("--generate-models/--dont-generate-models", default=True)
def main(generate_models):
    config = util.get_config()
    if generate_models:
        generate_models_for_hyper(config)
    generate_hyper_scores(config)


if __name__ == "__main__":
    main()
