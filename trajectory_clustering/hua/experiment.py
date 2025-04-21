from trajectory_clustering.hua.hua import Hua
from trajectory_clustering.experiment import (
    get_input,
    run_multiple_experiments,
    save_results,
)


if __name__ == "__main__":
    output_dir, D, bounds, n_runs, parallelize, n_cpus = get_input()

    phi = 100
    epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    ms = [40, 60, 80, 100]
    t_int = (0, D.shape[1] - 1)

    stats_dfs = []
    indiv_hd_dfs = []
    query_distortion_dfs = []

    M = lambda eps, m, tl, tu: Hua(m, phi, eps).publish(D[:, tl : tu + 1])

    id = 0
    for eps in epsilons:
        for m in ms:
            params = {"eps": eps, "m": m, "tl": t_int[0], "tu": t_int[1]}
            D_compare = D[:, t_int[0] : t_int[1] + 1]
            stats_df, indiv_hd_df, query_distortion_df = run_multiple_experiments(
                id, D_compare, bounds, M, params, n_runs, parallelize, n_cpus
            )
            stats_dfs.append(stats_df)
            indiv_hd_dfs.append(indiv_hd_df)
            query_distortion_dfs.append(query_distortion_df)
            id += 1

    eps = 2
    m = 60
    t_ints = [(0, tu) for tu in range(1, D.shape[1] + 1) if tu < 8]
    for t_int in t_ints:
        params = {"eps": eps, "m": m, "tl": t_int[0], "tu": t_int[1]}
        D_compare = D[:, t_int[0] : t_int[1] + 1]
        stats_df, indiv_hd_df, query_distortion_df = run_multiple_experiments(
            id, D_compare, bounds, M, params, n_runs, parallelize
        )
        stats_dfs.append(stats_df)
        indiv_hd_dfs.append(indiv_hd_df)
        query_distortion_dfs.append(query_distortion_df)
        id += 1

    save_results(output_dir, stats_dfs, indiv_hd_dfs, query_distortion_dfs)
