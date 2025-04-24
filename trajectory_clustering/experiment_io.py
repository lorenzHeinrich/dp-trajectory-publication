import os
import sys

import pandas as pd

from trajectory_clustering.data.read_db import csv_db_to_numpy


def get_input():
    args = sys.argv[1:]
    if len(args) < 1:
        raise ValueError("Please provide the path to the dataset.")
    if len(args) < 2:
        raise ValueError("Please provide the path to the output directory.")
    if len(args) < 3:
        raise ValueError("Plese provide the number of runs per configuration.")

    out_dir = args[1]
    n_runs = int(args[2])
    parallelize = "True" == args[3] if len(args) > 3 else True
    n_cpus = int(args[4]) if len(args) > 4 else -1
    df = pd.read_csv(args[0])
    D = csv_db_to_numpy(df)

    min_x, min_y = D[:, :, 0].min(), D[:, :, 1].min()
    max_x, max_y = D[:, :, 0].max(), D[:, :, 1].max()
    bounds = ((min_x - 0.1, max_x + 0.1), (min_y - 0.1, max_y + 0.1))

    return out_dir, D, bounds, n_runs, parallelize, n_cpus


def save_results(output_dir, stats_dfs, indiv_hd_dfs, query_distortion_dfs):
    """
    Save the results of the experiments to CSV files.
    :param output_dir: Directory to save the results
    :param stats_dfs: List of DataFrames containing statistics
    :param indiv_hd_dfs: List of DataFrames containing individual Hausdorff distances
    :param query_distortion_dfs: List of DataFrames containing query distortion results
    """
    os.makedirs(output_dir, exist_ok=True)
    stats_df = pd.concat(stats_dfs, ignore_index=True)
    indiv_hd_df = pd.concat(indiv_hd_dfs, ignore_index=True)
    query_distortion_df = pd.concat(query_distortion_dfs, ignore_index=True)

    stats_df.to_csv(os.path.join(output_dir, "stats.csv"), index=False)
    indiv_hd_df.to_csv(os.path.join(output_dir, "indiv_hd.csv"), index=False)
    query_distortion_df.to_csv(
        os.path.join(output_dir, "query_distortion.csv"), index=False
    )


def make_stats_df(id, run, param_dict, duration, size_unique, size_acc, hd):
    return pd.DataFrame(
        [
            {
                "id": id,
                "run": run,
                **param_dict,
                "duration": duration,
                "size_unique": size_unique,
                "size_acc": size_acc,
                "hausdorff": hd,
            }
        ]
    )


def make_indiv_hd_df(id, run, param_dict, indiv_hd_dists, counts):
    return pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "id": id,
                        "run": run + 1,
                        **param_dict,
                        "idx": i,
                        "individual_hausdorff": indiv_hd,
                        "count": counts[i],
                    }
                ]
            )
            for i, indiv_hd in enumerate(indiv_hd_dists)
        ],
        ignore_index=True,
    )


def make_query_distortion_df(id, run, param_dict, t_range, radii, distortions):
    return pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "id": id,
                        "run": run + 1,
                        **param_dict,
                        "query_run": i + 1,
                        "t_range": t_range,
                        "radius": r,
                        "psi_distortion_abs": psi_abs,
                        "psi_distortion_rel": psi_rel,
                        "dai_distortion_abs": dai_abs,
                        "dai_distortion_rel": dai_rel,
                    }
                ]
            )
            for i, (((psi_abs, psi_rel), (dai_abs, dai_rel)), r) in enumerate(
                zip(distortions, radii)
            )
        ],
        ignore_index=True,
    )
