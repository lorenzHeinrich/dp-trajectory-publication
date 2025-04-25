import json
import os
import sys

import numpy as np
import pandas as pd

from trajectory_clustering.data.read_db import csv_db_to_numpy


def get_input():
    args = sys.argv[1:]
    if len(args) < 1:
        raise ValueError("Please provide the path to the config file.")
    config_path = args[0]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    config = read_config(config_path)

    return config


def save_results(output_dir, stats_dfs, indiv_hd_dfs, query_distortion_dfs):
    """
    Save the results of the experiments to CSV files.

    Parameters:
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


def read_config(config_path="experiment_config/template.json"):
    with open(config_path) as f:
        cfg = json.load(f)

    # Load dataset
    df = pd.read_csv(cfg["dataset"])
    D = csv_db_to_numpy(df)

    # Compute bounds
    min_x, min_y = D[:, :, 0].min(), D[:, :, 1].min()
    max_x, max_y = D[:, :, 0].max(), D[:, :, 1].max()
    bounds = ((min_x - 0.1, max_x + 0.1), (min_y - 0.1, max_y + 0.1))

    # DPAPT config
    epsilons = cfg["dpapt"]["eps"]
    t_ints = [(0, tu) for tu in cfg["dpapt"]["tu"]]

    # AC config
    ac = cfg["adaptive_cells"]
    ac_config = ACConfig(
        cs=ac["c"],
        f_m1=ac["f_m1"],
        f_m2=ac["f_m2"],
        n_clusters=ac["n_clusters"],
        do_filter=ac["do_filter"],
    )

    # PP config
    pp = cfg["postprocess"]
    pp_config = PPConfig(sample=pp["sample"], uniform=pp["uniform"])

    # DPAPT config
    dpapt_config = DPAPTConfig(epsilons=epsilons, t_ints=t_ints)

    return ExperimentConfig(
        experiment_name=cfg["experiment_name"],
        output_dir=cfg["output_dir"],
        log_level=cfg.get("log_level"),
        D=D,
        bounds=bounds,
        n_runs=cfg["n_runs"],
        parallelize=cfg["parallelize"],
        n_cpus=cfg["n_cpus"],
        ac_config=ac_config,
        dpapt_config=dpapt_config,
        pp_config=pp_config,
    )


class ACConfig:
    def __init__(
        self,
        cs: list[float],
        f_m1: list[str],
        f_m2: list[str],
        n_clusters: list[int],
        do_filter: list[bool],
    ):
        self.cs = cs
        self.f_m1 = f_m1
        self.f_m2 = f_m2
        self.n_clusters = n_clusters
        self.do_filter = do_filter


class DPAPTConfig:
    def __init__(self, epsilons: list[float], t_ints: list[tuple[int, int]]):
        self.epsilons = epsilons
        self.t_ints = t_ints


class PPConfig:
    def __init__(self, sample: list[bool], uniform: list[bool]):
        self.sample = sample
        self.uniform = uniform


class ParameterConfig:
    def __init__(
        self,
        eps: float,
        t_int: tuple[int, int],
        n_clusters: int,
        do_filter: bool,
        sample: bool,
        uniform: bool,
        c: float,
        f_m1: str,
        f_m2: str,
    ):
        self.eps = eps
        self.t_int = t_int
        self.n_clusters = n_clusters
        self.do_filter = do_filter
        self.sample = sample
        self.uniform = uniform
        self.c = c
        self.f_m1 = f_m1
        self.f_m2 = f_m2

    def to_dict(self):
        return {
            "eps": self.eps,
            "tl": self.t_int[0],
            "tu": self.t_int[1],
            "n_clusters": self.n_clusters,
            "do_filter": self.do_filter,
            "sample": self.sample,
            "uniform": self.uniform,
            "c": self.c,
            "f_m1": self.f_m1,
            "f_m2": self.f_m2,
        }


class ExperimentConfig:
    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        log_level: str,
        D: np.ndarray,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        n_runs: int,
        parallelize: bool,
        n_cpus: int,
        ac_config: ACConfig,
        dpapt_config: DPAPTConfig,
        pp_config: PPConfig,
    ):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.log_level = log_level
        self.D = D
        self.bounds = bounds
        self.n_runs = n_runs
        self.parallelize = parallelize
        self.n_cpus = n_cpus
        self.ac_config = ac_config
        self.dpapt_config = dpapt_config
        self.pp_config = pp_config

    def get_param_combinations(self):
        param_combinations: list[ParameterConfig] = []
        for eps in self.dpapt_config.epsilons:
            for t_int in self.dpapt_config.t_ints:
                for n_clusters in self.ac_config.n_clusters:
                    for do_filter in self.ac_config.do_filter:
                        for sample in self.pp_config.sample:
                            for uniform in self.pp_config.uniform:
                                for c in self.ac_config.cs:
                                    for f_m1 in self.ac_config.f_m1:
                                        for f_m2 in self.ac_config.f_m2:
                                            param_combinations.append(
                                                ParameterConfig(
                                                    eps=eps,
                                                    t_int=t_int,
                                                    n_clusters=n_clusters,
                                                    do_filter=do_filter,
                                                    sample=sample,
                                                    uniform=uniform,
                                                    c=c,
                                                    f_m1=f_m1,
                                                    f_m2=f_m2,
                                                )
                                            )
        return param_combinations
