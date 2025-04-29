import logging

import numpy as np

from trajectory_clustering.data.read_db import t_drive
from trajectory_clustering.experiment.experiment import run_multiple_experiments
from trajectory_clustering.experiment.experiment_io import save_results
from trajectory_clustering.hua.hua import Hua

logger = logging.getLogger(__name__)


def Hua_wrapper(D):
    def M(ep, m):
        hua = Hua(m, 100, ep)
        D_pub, counts = hua.publish(D)
        D_exp = np.concatenate(
            [
                np.tile(D_pub[i][np.newaxis, :, :], (counts[i], 1, 1))
                for i in range(len(D_pub))
            ],
            axis=0,
        )
        return D_exp, np.zeros((D_exp.shape[0], D_exp.shape[1]))

    return lambda ep, m, tl, tu: M(ep, m)


if __name__ == "__main__":

    logger.setLevel(logging.INFO)
    logger.info(f"Running Hua experiment")

    stats_dfs, indiv_hd_dfs, query_distortion_dfs = [], [], []

    D, bounds = t_drive("medium")
    eps = [1.0, 2.0]
    ms = [15]
    t_ints = [(0, 3), (0, 7), (0, 15)]

    param_combinations = [
        {"ep": ep, "m": m, "tl": t_int[0], "tu": t_int[1]}
        for ep in eps
        for m in ms
        for t_int in t_ints
    ]

    for id, combination in enumerate(param_combinations):
        logger.info(f"Running parameter combination {id + 1}: {combination}")
        D_in = D.copy()[:, combination["tl"] : combination["tu"] + 1]
        M = Hua_wrapper(D_in)

        stats, indiv_hd, query_dist = run_multiple_experiments(
            id=id + 1,
            D=D_in,
            bounds=bounds,
            M=M,
            params=combination,
            n_runs=16,
            parallelize=True,
            n_cpus=-1,
        )
        stats_dfs.append(stats)
        indiv_hd_dfs.append(indiv_hd)
        query_distortion_dfs.append(query_dist)

    save_results("results/hua/medium", stats_dfs, indiv_hd_dfs, query_distortion_dfs)
