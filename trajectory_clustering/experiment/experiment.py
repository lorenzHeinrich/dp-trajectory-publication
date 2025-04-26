import logging
import numpy as np
import pandas as pd

from time import time
from joblib import Parallel, delayed

from trajectory_clustering.experiment.experiment_io import (
    make_indiv_hd_df,
    make_query_distortion_df,
    make_stats_df,
)
from trajectory_clustering.experiment.metrics import (
    hausdorff,
    individual_hausdorff,
    query_distortion,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_multiple_experiments(
    id, D, bounds, M, params, n_runs=16, parallelize=True, n_cpus=-1
):
    results: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    if parallelize:
        results = Parallel(n_jobs=n_cpus)(  # type: ignore
            delayed(experiment)(id, run, D, bounds, M, params) for run in range(n_runs)
        )

    else:
        for run in range(n_runs):
            result = experiment(id, run, D, bounds, M, params)
            results.append(result)
    logger.info(f"All runs completed for experiment {id}.")

    stats_dfs = [result[0] for result in results]

    indiv_hd_dfs = [result[1] for result in results]
    query_distortion_dfs = [result[2] for result in results]

    stats_df = pd.concat(stats_dfs, ignore_index=True)
    indiv_hd_df = pd.concat(indiv_hd_dfs, ignore_index=True)
    query_distortion_df = pd.concat(query_distortion_dfs, ignore_index=True)

    return stats_df, indiv_hd_df, query_distortion_df


def experiment(id, run, D, bounds, M, params):
    logger.info(f"Running run {run + 1} for experiment {id} with params: {params}")
    D = D.copy()

    start = time()
    D_pub, U = M(**params)
    duration = time() - start

    if D_pub.shape[0] == 0:
        logger.warning(
            f"Published dataset is empty for run {run + 1} of experiment {id}."
        )
        stats_df = make_stats_df(id, run, params, duration, 0, 0, np.nan)
        return stats_df, pd.DataFrame(), pd.DataFrame()

    logger.info(
        f"Run {run + 1} of experiment {id} completed in {duration:.2f} seconds."
    )
    logger.info(f"Published dataset shape: {D_pub.shape}")

    D_pub_unique, counts = np.unique(
        D_pub.reshape(D_pub.shape[0], -1), axis=0, return_counts=True
    )
    D_pub_unique = D_pub_unique.reshape(D_pub_unique.shape[0], D_pub.shape[1], 2)

    hd = hausdorff(D, D_pub_unique)
    stats_df = make_stats_df(id, run, params, duration, len(counts), np.sum(counts), hd)

    indiv_hd_dists = individual_hausdorff(D, D_pub_unique)
    indiv_hd_df = make_indiv_hd_df(id, run, params, indiv_hd_dists, counts)

    t_range = np.arange(0, min(4, D_pub.shape[1] + 1))
    t_ints = [(tl, tu) for tl in t_range for tu in t_range if tl <= tu]
    n_queries = 50
    query_distortion_dfs = []

    for t_int in t_ints:
        distortions = []
        radii = []
        for _ in range(n_queries):
            (p, r) = random_region(bounds)
            radii.append(r)
            distortions.append(query_distortion(D, D_pub, (p, r), t_int, U))
        query_distortion_dfs.append(
            make_query_distortion_df(
                id,
                run,
                params,
                t_int[1] - t_int[0] + 1,
                radii,
                distortions,
            )
        )

    query_distortion_df = pd.concat(query_distortion_dfs, ignore_index=True)
    return stats_df, indiv_hd_df, query_distortion_df


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
        np.random.choice([span // 20, span // 40, span // 80]),
    )
