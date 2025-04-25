import logging
import numpy as np
from trajectory_clustering.dpapt.adaptive_cells import AdaptiveCells
from trajectory_clustering.dpapt.dpapt import DPAPT
from trajectory_clustering.dpapt.post_process import post_process_with_uncertainty
from trajectory_clustering.experiment.experiment import run_multiple_experiments
from trajectory_clustering.experiment.experiment_io import get_input, save_results

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_fm1(f_m1):
    if f_m1 == "eps_sensitive":
        return lambda N, eps, c: max(
            10, int(1 / 4 * np.ceil(np.ceil(np.sqrt(N * eps / c))))
        )
    if f_m1 == "eps_agnostic":
        return lambda N, _, c: max(5, int(1 / 4 * np.ceil(np.ceil(np.sqrt(N / c)))))


def get_fm2(f_m2):
    if f_m2 == "eps_sensitive":
        return lambda nc, eps, c: int(np.ceil(np.sqrt(max(1, nc * eps) / (c / 2))))
    if f_m2 == "eps_agnostic":
        return lambda nc, _, c: int(np.ceil(np.sqrt(max(1, nc) / (c / 2))))


def DPAPT_wrapper(D, bounds):
    def M(eps, t_int, n_clusters, do_filter, sample, uniform, c, f_m1, f_m2):
        ac = AdaptiveCells(
            c=c,
            f_m1=get_fm1(f_m1),
            f_m2=get_fm2(f_m2),
            do_filter=do_filter,
            n_clusters=n_clusters,
        )
        D_areas, counts = DPAPT(ac=ac).publish(D, t_int, bounds, eps)
        D_post, U = post_process_with_uncertainty(D_areas, counts, sample, uniform)

        return D_post, U

    return lambda eps, tl, tu, n_clusters, do_filter, sample, uniform, c, f_m1, f_m2: M(
        eps, (tl, tu), n_clusters, do_filter, sample, uniform, c, f_m1, f_m2
    )


if __name__ == "__main__":

    config = get_input()
    logger.setLevel(config.log_level)
    logger.info(f"Running DPAPT experiment {config.experiment_name}")

    param_cobinations = config.get_param_combinations()
    logger.info(f"Number of parameter combinations: {len(param_cobinations)}")

    stats_dfs, indiv_hd_dfs, query_distortion_dfs = [], [], []

    for id, combination in enumerate(param_cobinations):
        params = combination.to_dict()
        logger.info(f"Running parameter combination {id + 1}: {params}")
        M = DPAPT_wrapper(config.D, config.bounds)
        D_compare = config.D[:, combination.t_int[0] : combination.t_int[1] + 1]

        stats, indiv_hd, query_dist = run_multiple_experiments(
            id=id + 1,
            D=D_compare,
            bounds=config.bounds,
            M=M,
            params=params,
            n_runs=config.n_runs,
            parallelize=config.parallelize,
            n_cpus=config.n_cpus,
        )
        stats_dfs.append(stats)
        indiv_hd_dfs.append(indiv_hd)
        query_distortion_dfs.append(query_dist)

    save_results(config.output_dir, stats_dfs, indiv_hd_dfs, query_distortion_dfs)
