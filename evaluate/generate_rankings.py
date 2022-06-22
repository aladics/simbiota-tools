from pathlib import Path
import json
from math import isclose
from typing import Dict, List

import click

from common import util


def sort_dict(dict_, reverse=False):
    """Sort dict by value."""
    return {
        k: v
        for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=reverse)
    }


def get_rank_score(val: float, vals: List[float], reverse: bool) -> int:
    """
    Get the rank score for a model.

    Calculated as (n - 1) - k where:
    n: number of models
    k: other models that achieve better score.

    Best score is n - 1, worst score is 0.

    If :param reverse is True, the lower values mean better score, otherwise higher values mean better score.
    """
    n_models = len(util.get_model_names())

    if not reverse:
        k = sum(val_ > val for val_ in vals if not isclose(val, val_, rel_tol=1e-14))
    else:
        k = sum(val_ < val for val_ in vals if not isclose(val, val_, rel_tol=1e-14))

    return (n_models - 1) - k


def get_ranks(metric_scores: Dict[str, float], reverse: bool = False) -> Dict[str, int]:
    ranks = {}
    keys = list(metric_scores.keys())
    vals = list(metric_scores.values())
    for idx in range(len(vals)):
        rank = get_rank_score(vals[idx], vals, reverse)
        ranks[keys[idx]] = rank

    return ranks


def get_ranks_by_weeks(runs_data, n_run, arch, model_name, metric):
    res = []
    for n_week in util.get_weeks_range():
        res.append(runs_data[n_run][n_week][arch]["ranks"][metric][model_name])

    week_count = len(util.get_weeks_range())
    all_models_rank_sum = week_count * (len(util.get_model_names()) - 1)
    current_model_rank_sum = sum(res)
    return (
        res,
        current_model_rank_sum / week_count,
        current_model_rank_sum / all_models_rank_sum,
    )


def get_aggregated_ranks_by_weeks(runs_data, n_run, arch, model_name):
    prec_ranks, avg_prec, norm_prec = get_ranks_by_weeks(
        runs_data, n_run, arch, model_name, "precision"
    )
    recall_ranks, avg_recall, norm_recall = get_ranks_by_weeks(
        runs_data, n_run, arch, model_name, "recall"
    )
    memory_ranks, avg_memory, norm_memory = get_ranks_by_weeks(
        runs_data, n_run, arch, model_name, "memory"
    )
    time_ranks, avg_time, norm_time = get_ranks_by_weeks(
        runs_data, n_run, arch, model_name, "time"
    )

    res = {
        "precision": {"raw": prec_ranks, "average": avg_prec, "normalized": norm_prec},
        "recall": {
            "raw": recall_ranks,
            "average": avg_recall,
            "normalized": norm_recall,
        },
        "memory": {
            "raw": memory_ranks,
            "average": avg_memory,
            "normalized": norm_memory,
        },
        "time": {"raw": time_ranks, "average": avg_time, "normalized": norm_time},
    }

    return res


def get_utility_score(model_rank_scores, usecase: str, aggr_type: str):
    usecase_weights = util.get_usecase_weights(usecase)
    prec_score = usecase_weights["alpha"] * model_rank_scores["precision"][aggr_type]
    recall_score = usecase_weights["beta"] * model_rank_scores["recall"][aggr_type]
    memory_score = usecase_weights["gamma"] * model_rank_scores["memory"][aggr_type]
    time_score = usecase_weights["delta"] * model_rank_scores["time"][aggr_type]

    return prec_score + recall_score + memory_score + time_score


def get_ranks_by_model(runs_data, usecase):
    model_ranks = {}
    for n_run in util.get_runs_range():
        model_ranks[n_run] = {}
        for model_name in util.get_model_names():
            model_ranks[n_run][model_name] = {}
            for arch in util.get_arch_types():
                model_ranks[n_run][model_name][arch] = {}
                model_ranks[n_run][model_name][arch][
                    "ranks"
                ] = get_aggregated_ranks_by_weeks(runs_data, n_run, arch, model_name)
                model_ranks[n_run][model_name][arch][
                    "utility_score"
                ] = get_utility_score(
                    model_ranks[n_run][model_name][arch]["ranks"], usecase, "normalized"
                )

    return model_ranks


def get_metric_csv_header():
    header = ["model_name"]
    for n_week in util.get_weeks_range():
        header.append(str(n_week))

    header += ["average", "normalized"]
    return header


def create_csv_line(list_):
    return ",".join(list_) + "\n"


def append_str_to_csv(str: str, csv_path: Path):
    with csv_path.open("a") as f:
        f.write(str)


def dump_metric(ranks_by_model, metric_name, arch, root_path):
    metric_path = root_path / f"{metric_name}.csv"
    if metric_path.exists():
        metric_path.unlink()

    header = create_csv_line(get_metric_csv_header())
    append_str_to_csv(header, metric_path)

    for model_name in util.get_model_names():
        curr_line = [model_name]
        metric_stats = ranks_by_model[model_name][arch]["ranks"][metric_name]
        curr_line += [str(metric_val) for metric_val in metric_stats["raw"]]
        curr_line += [str(metric_stats["average"]), str(metric_stats["normalized"])]
        append_str_to_csv(create_csv_line(curr_line), metric_path)


def get_rank_csv_header():
    header = ["rank", "model_name"]
    header += util.get_metric_names()
    header.append("utility_score")
    return header


def dump_ranks(sorted_models, arch, root_path):
    ranks_path = root_path / "ranks.csv"
    if ranks_path.exists():
        ranks_path.unlink()

    append_str_to_csv(create_csv_line(get_rank_csv_header()), ranks_path)

    rank = 1
    for model_name, model_val in sorted_models:
        model_stats = model_val[arch]
        curr_line = [str(rank), model_name]
        for metric_name in util.get_metric_names():
            curr_line.append(str(model_stats["ranks"][metric_name]["normalized"]))
        curr_line.append(str(model_stats["utility_score"]))
        rank += 1

        append_str_to_csv(create_csv_line(curr_line), ranks_path)


def dump_csvs(ranks_by_model):
    for n_run in util.get_runs_range():
        for arch in util.get_arch_types():
            ranks_run_dir = util.get_ranks_run_dir(n_run, arch)
            ranks_run_dir.mkdir(exist_ok=True, parents=True)
            sorted_models = sorted(
                ranks_by_model[n_run].items(),
                key=lambda item: item[1][arch]["utility_score"],
                reverse=True,
            )
            dump_ranks(sorted_models, arch, ranks_run_dir)

            for metric_name in util.get_metric_names():
                dump_metric(ranks_by_model[n_run], metric_name, arch, ranks_run_dir)


@click.command()
@click.option(
    "--usecase-type",
    help="The usecase type that correspond to the metaparams (alpha, beta etc.)",
    required=True,
)
def run(usecase_type: str):

    runs_data = {}
    for n_run in util.get_runs_range():
        runs_data[n_run] = {}
        for n_week in util.get_weeks_range():
            runs_data[n_run][n_week] = {}
            for arch in util.get_arch_types():
                prec_scores = {}
                recall_scores = {}
                memory_scores = {}
                time_scores = {}
                task_id = util.generate_task_id(arch, n_run, n_week)
                curr_path = (
                    util.get_eval_results_path() / str(n_run) / f"{task_id}.json"
                )
                with curr_path.open("r") as ranks_fp:
                    curr_result = json.load(ranks_fp)
                    for model in curr_result:
                        model_name = model["strategy"].split(" ")[0]
                        prec_scores[model_name] = model["test"]["precision"]
                        recall_scores[model_name] = model["test"]["recall"]
                        memory_scores[model_name] = model["memory"]
                        time_scores[model_name] = model["prediction time"]

                runs_data[n_run][n_week][arch] = {}
                runs_data[n_run][n_week][arch]["scores"] = {}
                runs_data[n_run][n_week][arch]["scores"]["precision"] = sort_dict(
                    prec_scores
                )
                runs_data[n_run][n_week][arch]["scores"]["recall"] = sort_dict(
                    recall_scores
                )
                runs_data[n_run][n_week][arch]["scores"]["memory"] = sort_dict(
                    memory_scores, True
                )
                runs_data[n_run][n_week][arch]["scores"]["time"] = sort_dict(
                    time_scores, True
                )

                runs_data[n_run][n_week][arch]["ranks"] = {}
                runs_data[n_run][n_week][arch]["ranks"]["precision"] = get_ranks(
                    runs_data[n_run][n_week][arch]["scores"]["precision"]
                )
                runs_data[n_run][n_week][arch]["ranks"]["recall"] = get_ranks(
                    runs_data[n_run][n_week][arch]["scores"]["recall"]
                )
                runs_data[n_run][n_week][arch]["ranks"]["memory"] = get_ranks(
                    runs_data[n_run][n_week][arch]["scores"]["memory"], reverse=True
                )
                runs_data[n_run][n_week][arch]["ranks"]["time"] = get_ranks(
                    runs_data[n_run][n_week][arch]["scores"]["time"], reverse=True
                )

    util.get_eval_ranks_path().mkdir(parents=True, exist_ok=True)
    with util.get_ranks_data_path().open("w") as ranks_fp:
        json.dump(runs_data, ranks_fp, indent=2)

    ranks_by_model = get_ranks_by_model(runs_data, usecase_type)
    dump_csvs(ranks_by_model)


if __name__ == "__main__":
    run()
