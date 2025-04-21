import logging
import os
import sys
import numpy as np
import pandas as pd

from time import time
from joblib import Parallel, delayed

from trajectory_clustering.data.read_db import csv_db_to_numpy
from trajectory_clustering.metrics import hausdorff, indiv_hausdorff, query_distortion

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


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


def run_multiple_experiments(
    id, D, bounds, M, params, n_runs=16, parallelize=True, n_cpus=-1
):
    results = []
    if parallelize:
        results = Parallel(n_jobs=n_cpus)(
            delayed(experiment)(id, run, D, bounds, M, params) for run in range(n_runs)
        )
    else:
        for run in range(n_runs):
            result = experiment(id, run, D, bounds, M, params)
            results.append(result)
    logger.info(f"All runs completed for experiment {id}.")

    # collect results
    stats_dfs = [result[0] for result in results]  # type: ignore
    indiv_hd_dfs = [result[1] for result in results]  # type: ignore
    query_distortion_dfs = [result[2] for result in results]  # type: ignore

    stats_df = pd.concat(stats_dfs, ignore_index=True)
    indiv_hd_df = pd.concat(indiv_hd_dfs, ignore_index=True)
    query_distortion_df = pd.concat(query_distortion_dfs, ignore_index=True)

    return stats_df, indiv_hd_df, query_distortion_df


def experiment(id, run, D, bounds, M, params):
    logger.info(f"Running run {run + 1} for experiment {id} with params: {params}")
    start = time()
    D = D.copy()
    D_pub, counts = M(**params)
    duration = time() - start

    # the mechaism did not publish anything, possibly due to a too small eps
    if D_pub.shape[0] == 0:
        logger.warning(
            f"Published dataset is empty for run {run + 1} of experiment {id}."
        )
        stats_df = make_stats_df(id, run, params, duration, 0, 0, np.nan)
        return stats_df, pd.DataFrame(), pd.DataFrame()

    D_exp = expand(D_pub, counts)
    hd = hausdorff(D, D_exp)
    stats_df = make_stats_df(
        id, run, params, duration, D_pub.shape[0], np.sum(counts), hd
    )

    indiv_hd_dists = indiv_hausdorff(D, D_exp)
    indiv_hd_df = make_indiv_hd_df(id, run, params, indiv_hd_dists)

    t_range = np.arange(0, D.shape[1] + 1, step=max(1, D.shape[1] // 4))
    t_ints = [(tl, tu) for tl in t_range for tu in t_range if tl < tu]
    uncertainties = [2, 10, 25]
    n_queries = 50
    query_distortion_dfs = []
    for uncertainty in uncertainties:
        for t_int in t_ints:
            distortions = []
            for _ in range(n_queries):
                R = random_region(bounds)
                distortions.append(query_distortion(D, D_exp, R, t_int, uncertainty))
            query_distortion_dfs.append(
                make_query_distortion_df(
                    id,
                    run,
                    params,
                    uncertainty,
                    t_int[1] - t_int[0],
                    distortions,
                )
            )
    query_distortion_df = pd.concat(query_distortion_dfs, ignore_index=True)
    return (stats_df, indiv_hd_df, query_distortion_df)


def random_region(bounds):
    """
    Generate a random region within the given bounds.
    :param bounds: Bounds for the region
    :return: Random region
    """
    span = bounds[1][1] - bounds[1][0]
    return (
        (
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
        ),
        np.random.uniform(span // 100, span // 10),
    )


def expand(D, counts):
    """
    Expand the dataset D based on the counts of each trajectory.

    :param D: Dataset to expand
    :param counts: Counts of each trajectory
    :return: Expanded dataset
    """
    return np.concatenate(
        [np.repeat([T], count, axis=0) for T, count in zip(D, counts)]
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


def make_indiv_hd_df(id, run, param_dict, indiv_hd_dists):
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
                    }
                ]
            )
            for i, indiv_hd in enumerate(indiv_hd_dists)
        ],
        ignore_index=True,
    )


def make_query_distortion_df(id, run, param_dict, uncertainty, t_range, distortions):
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
                        "uncertainty": uncertainty,
                        "psi_distortion": psi_distortion,
                        "dai_distortion": dai_distortion,
                    }
                ]
            )
            for i, (psi_distortion, dai_distortion) in enumerate(distortions)
        ],
        ignore_index=True,
    )
