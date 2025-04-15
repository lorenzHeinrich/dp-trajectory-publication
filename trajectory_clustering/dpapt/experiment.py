import os
from time import time
from turtle import pos
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from trajectory_clustering.data.read_db import csv_db_to_numpy
from trajectory_clustering.dpapt.dpapt import (
    DPAPT,
    post_process_centroid,
    post_process_uniform,
)
from trajectory_clustering.metrics import (
    expand,
    hausdorff,
    indiv_hausdorff,
    query_distortion,
)


def make_stats_df(
    run, eps, t_int, duration, unique_size, acc_size, hd, post_processing
):
    return pd.DataFrame(
        [
            {
                "run": run + 1,
                "eps": eps,
                "tl": t_int[0],
                "tu": t_int[1],
                "post_processing": post_processing,
                "duration": duration,
                "unique_size": unique_size,
                "acc_size": acc_size,
                "hausdorff": hd,
            }
        ]
    )


def make_indiv_hd_df(run, eps, t_int, indiv_hd_dists, post_processing):
    return pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "run": run + 1,
                        "eps": eps,
                        "tl": t_int[0],
                        "tu": t_int[1],
                        "post_processing": post_processing,
                        "idx": i,
                        "individual_hausdorff": indiv_hd,
                    }
                ]
            )
            for i, indiv_hd in enumerate(indiv_hd_dists)
        ],
        ignore_index=True,
    )


def make_query_distortion_df(run, eps, t_int, query_distortion_dists, post_processing):
    return pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "run": run + 1,
                        "eps": eps,
                        "tl": t_int[0],
                        "tu": t_int[1],
                        "post_processing": post_processing,
                        "query_run": i + 1,
                        "psi_distortion": psi_distortion,
                        "dai_distortion": dai_distortion,
                    }
                ]
            )
            for i, (psi_distortion, dai_distortion) in enumerate(query_distortion_dists)
        ],
        ignore_index=True,
    )


def run(run, D, bounds, eps, t_int, post_processing, name):
    print(f"Run {run + 1}: eps = {eps}, t_int = {t_int}")
    dpapt = DPAPT(
        alpha=0.5,
        beta=0.5,
        gamma=0.1,
        c1=10,
    )

    start = time()
    D_cells, counts = dpapt.publish(D, t_int, bounds, eps)
    duration = time() - start
    if D_cells.shape[0] == 0:
        stats_df = make_stats_df(run, eps, t_int, duration, 0, 0, None, None)
        return stats_df, pd.DataFrame(), pd.DataFrame()

    D_pub = post_processing(D_cells)
    D_expanded = expand(D_pub, counts)
    D_trunc = D[:, t_int[0] : t_int[1] + 1]
    hd = hausdorff(D_trunc, D_expanded)
    stats_df = make_stats_df(
        run, eps, t_int, duration, len(counts), np.sum(counts), hd, name
    )

    indiv_hd_dists = indiv_hausdorff(D_trunc, D_expanded)
    indiv_hd_df = make_indiv_hd_df(run, eps, t_int, indiv_hd_dists, name)

    distortions = np.array(
        [query_distortion(D_trunc, D_expanded, bounds, 10) for _ in range(100)]
    )
    query_distortion_df = make_query_distortion_df(run, eps, t_int, distortions, name)

    return (stats_df, indiv_hd_df, query_distortion_df)


def run_multiple(D, bounds, times, eps, t_int, post_processing, name):
    """
    Run multiple experiments in parallel.

    :param D: Dataset
    :param bounds: Bounds for the dataset
    :param times: Number of runs
    :param m: Parameter m
    :param eps: Epsilon value
    :param t_int: Time interval
    """
    results = Parallel(n_jobs=-1)(
        delayed(run)(i, D, bounds, eps, t_int, post_processing, name)
        for i in range(times)
    )
    print("All runs completed.")
    stats_dfs = [result[0] for result in results]  # type: ignore
    indiv_hd_dfs = [result[1] for result in results]  # type: ignore
    query_distortion_dfs = [result[2] for result in results]  # type: ignore

    # write to csv
    stats_df = pd.concat(stats_dfs, ignore_index=True)
    indiv_hd_df = pd.concat(indiv_hd_dfs, ignore_index=True)
    query_distortion_df = pd.concat(query_distortion_dfs, ignore_index=True)

    return stats_df, indiv_hd_df, query_distortion_df


if __name__ == "__main__":
    t_drive = pd.read_csv(
        "t-drive-trajectories/release/taxi_log_2008_by_id/cleaned_normalized.csv"
    )
    sample = t_drive[t_drive["date"] == "2008-02-03"]
    D = csv_db_to_numpy(sample)
    bounds = ((0, 100), (0, 100))
    runs = 8

    os.makedirs("results/dpapt", exist_ok=True)
    stats_dfs = []
    indiv_hd_dfs = []
    query_distortion_dfs = []
    # vary eps amd t_int
    epsilons = [0.5, 1, 1.5, 2, 5]
    t_ints = [(0, i) for i in range(1, 10)]
    for eps in epsilons:
        for t_int in t_ints:
            for post_processing, name in [
                (post_process_centroid, "post_centroid"),
                (post_process_uniform, "post_uniform"),
            ]:
                stats, ihd, qd = run_multiple(
                    D, bounds, runs, eps, t_int, post_processing, name
                )
                stats_dfs.append(stats)
                indiv_hd_dfs.append(ihd)
                query_distortion_dfs.append(qd)

    # save results
    make_stats_df = pd.concat(stats_dfs, ignore_index=True)
    indiv_hd_df = pd.concat(indiv_hd_dfs, ignore_index=True)
    query_distortion_df = pd.concat(query_distortion_dfs, ignore_index=True)
    make_stats_df.to_csv("results/dpapt/stats.csv", index=False)
    indiv_hd_df.to_csv("results/dpapt/indiv_hd.csv", index=False)
    query_distortion_df.to_csv("results/dpapt/query_distortion.csv", index=False)
