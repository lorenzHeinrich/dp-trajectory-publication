from datetime import datetime
import os
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from time import time

from trajectory_clustering.hua.hua import Hua
from trajectory_clustering.data.read_db import csv_db_to_numpy, merge_t_drive_days
from trajectory_clustering.metrics import (
    expand,
    hausdorff,
    indiv_hausdorff,
    query_distortion,
)


def make_stats_df(run, eps, m, t_int, duration, size_unique, size_acc, hd):
    return pd.DataFrame(
        [
            {
                "run": run + 1,
                "eps": eps,
                "m": m,
                "tl": t_int[0],
                "tu": t_int[1],
                "duration": duration,
                "size_unique": size_unique,
                "size_acc": size_acc,
                "hausdorff": hd,
            }
        ]
    )


def make_indiv_hd_df(run, eps, m, t_int, indiv_hd_dists):
    return pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "run": run + 1,
                        "eps": eps,
                        "m": m,
                        "tl": t_int[0],
                        "tu": t_int[1],
                        "idx": i,
                        "individual_hausdorff": indiv_hd,
                    }
                ]
            )
            for i, indiv_hd in enumerate(indiv_hd_dists)
        ],
        ignore_index=True,
    )


def make_query_distortion_df(run, eps, m, t_int, distortions):
    return pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "run": run + 1,
                        "eps": eps,
                        "m": m,
                        "tl": t_int[0],
                        "tu": t_int[1],
                        "query_run": i + 1,
                        "psi_distortion": psi_distortion,
                        "dai_distortion": dai_distortion,
                    }
                ]
            )
            for i, (psi_distortion, dai_distortion) in enumerate(distortions)
        ],
        ignore_index=True,
    )


def run(run, D, bounds, m, eps, t_int):
    print(f"Run {run + 1}: m = {m}, eps = {eps}, t_int = {t_int}")
    phi = 20
    hua = Hua(m, phi, eps)
    D = D[:, t_int[0] : t_int[1] + 1]

    start = time()
    D_pub, counts = hua.publish(D)
    duration = time() - start
    D_expanded = expand(D_pub, counts)
    hd = hausdorff(D, D_expanded)
    stats_df = make_stats_df(
        run, eps, m, t_int, duration, len(counts), np.sum(counts), hd
    )

    indiv_hd_dists = indiv_hausdorff(D, D_expanded)
    indiv_hd_df = make_indiv_hd_df(run, eps, m, t_int, indiv_hd_dists)
    distortions = np.array(
        [query_distortion(D, D_expanded, bounds, 10) for _ in range(100)]
    )
    query_distortion_df = make_query_distortion_df(run, eps, m, t_int, distortions)

    return (stats_df, indiv_hd_df, query_distortion_df)


def run_multiple(D, bounds, times, m, eps, t_int, parallel=True):
    """
    Run multiple experiments in parallel.

    :param D: Dataset
    :param bounds: Bounds for the dataset
    :param times: Number of runs
    :param m: Parameter m
    :param eps: Epsilon value
    :param t_int: Time interval
    """
    results = []
    if parallel:
        results = Parallel(n_jobs=-1)(
            delayed(run)(i, D, bounds, m, eps, t_int) for i in range(times)
        )
    else:
        for i in range(times):
            result = run(i, D, bounds, m, eps, t_int)
            results.append(result)
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
    merged_days = merge_t_drive_days(t_drive, 2)
    print(merged_days.info())
    D = csv_db_to_numpy(merged_days)[:, :2]
    print(D.shape)
    bounds = ((0, 100), (0, 100))
    runs = 1

    run_id = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    dir_path = f"results/hua/{run_id}"

    os.makedirs(dir_path)
    stats_dfs = []
    indiv_hd_dfs = []
    query_distortion_dfs = []
    # vary eps and m for fixed t_int
    epsilons = [0.5, 1, 2, 5]
    ms = [20, 40, 60, 80]
    t_int = (0, D.shape[1] - 1)
    for eps in epsilons:
        for m in ms:
            stats, ihd, qd = run_multiple(D, bounds, runs, m, eps, t_int, False)
            stats_dfs.append(stats)
            indiv_hd_dfs.append(ihd)
            query_distortion_dfs.append(qd)

    # vary t_int for fixed m and eps
    eps = 1
    m = 60
    t_ints = [(0, i) for i in range(1, 10)] + [
        (0, D.shape[1] // 2),
        (0, D.shape[1] - 1),
    ]
    for t_int in t_ints:
        stats, ihd, qd = run_multiple(D, bounds, runs, m, eps, t_int)
        stats_dfs.append(stats)
        indiv_hd_dfs.append(ihd)
        query_distortion_dfs.append(qd)
    # save results
    stats_df = pd.concat(stats_dfs, ignore_index=True)
    indiv_hd_df = pd.concat(indiv_hd_dfs, ignore_index=True)
    query_distortion_df = pd.concat(query_distortion_dfs, ignore_index=True)
    stats_df.to_csv(f"{dir_path}/stats.csv", index=False)
    indiv_hd_df.to_csv(f"{dir_path}/indiv_hd.csv", index=False)
    query_distortion_df.to_csv(f"{dir_path}/distortion.csv", index=False)
