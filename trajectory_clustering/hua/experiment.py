import numpy as np
import pandas as pd

from matplotlib import gridspec, pyplot as plt
from joblib import Parallel, delayed

from trajectory_clustering.hua.hua import Hua
from trajectory_clustering.data.read_db import csv_db_to_numpy
from trajectory_clustering.metrics import (
    definitely_always_inside,
    hausdorff,
    indiv_hausdorff,
    possibly_sometimes_inside,
    range_query_distortion,
)


def run_experiment(D, eps, m, phi, i, j):
    print(f"Experiment {i + 1} - {j + 1}: m = {m}, eps = {eps}")
    hua = Hua(m, phi, eps)
    D_pub, counts = hua.publish(D)
    hd = hausdorff(D, D_pub)

    D_pub_extended = np.concatenate(
        [np.repeat([T], count, axis=0) for T, count in zip(D_pub, counts)]
    )
    indiv_hd = indiv_hausdorff(D, D_pub_extended)

    psi_distortions = np.empty((1000, 2))
    dai_distortions = np.empty((1000, 2))
    for run in range(1000):
        R = np.random.uniform(0, 100, 2)
        t_interval = np.random.randint(0, D.shape[1] - 1, 2)
        for radius, r in enumerate([0.5, 1]):
            q1 = lambda D: possibly_sometimes_inside(D, (R, r), t_interval, 0.5)
            q2 = lambda D: definitely_always_inside(D, (R, r), t_interval, 0.5)
            psi_distortion = range_query_distortion(D, D_pub_extended, q1)
            dai_distortion = range_query_distortion(D, D_pub_extended, q2)
            psi_distortions[run, radius] = psi_distortion
            dai_distortions[run, radius] = dai_distortion
    psi_distortions = np.mean(psi_distortions, axis=0)
    dai_distortions = np.mean(dai_distortions, axis=0)

    return (i, j, hd, indiv_hd, psi_distortions, dai_distortions)


def plot_results(epsilons, ms, results):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Hausdorff Distances and Distortions")

    gs = gridspec.GridSpec(4, 2, figure=fig)

    hd_ax = fig.add_subplot(gs[0, :])
    hd_ax.set_box_aspect(1)
    hd_ax.set_title("Hausdorff Distances")
    plot_hausdorff(epsilons, ms, results, hd_ax)

    indiv_hd_ax1 = fig.add_subplot(gs[1, 0])
    indiv_hd_ax1.set_box_aspect(1)
    plot_indiv_hd_var_eps(epsilons, ms, len(ms) // 2, results, indiv_hd_ax1)

    indiv_hd_ax2 = fig.add_subplot(gs[1, 1])
    indiv_hd_ax2.set_box_aspect(1)
    plot_indiv_hd_var_m(epsilons, len(epsilons) // 2, ms, results, indiv_hd_ax2)

    query_distortion_axq11 = fig.add_subplot(gs[2, 0], projection="3d")
    plot_query_distortion(
        epsilons, ms, "PSI Query", 2, 0.5, 0, results, query_distortion_axq11
    )
    query_distortion_axq12 = fig.add_subplot(gs[2, 1], projection="3d")
    plot_query_distortion(
        epsilons, ms, "PSI Query", 2, 1, 1, results, query_distortion_axq12
    )
    query_distortion_axdai1 = fig.add_subplot(gs[3, 0], projection="3d")
    plot_query_distortion(
        epsilons, ms, "DAI Query", 3, 0.5, 0, results, query_distortion_axdai1
    )
    query_distortion_axdai2 = fig.add_subplot(gs[3, 1], projection="3d")
    plot_query_distortion(
        epsilons, ms, "DAI Query", 3, 1, 1, results, query_distortion_axdai2
    )

    plt.tight_layout()
    plt.show()


def plot_hausdorff(epsilons, ms, results, ax):
    for i, eps in enumerate(epsilons):
        hd = np.array([results[i, j][0] for j in range(len(ms))])
        ax.plot(ms, hd, marker="o", label=r"$\varepsilon=" + str(eps) + r"$")
    ax.legend()
    ax.set_xlabel("m")
    ax.set_ylabel("Hausdorff Distance")
    ax.set_title("Hausdorff Distance vs m")


def plot_indiv_hd_var_m(epsilons, ideps, ms, results, ax):
    step = 10
    for i, m in enumerate(ms):
        indiv_hd = results[ideps, i][1]
        counts = np.histogram(indiv_hd, bins=step)[0]
        cdf = np.cumsum(counts) / len(indiv_hd)
        ax.plot(np.arange(step), cdf, label=r"$m=" + str(m) + r"$")
    ax.legend()
    ax.set_xlabel("Indiv Hausdorff Distance")
    ax.set_ylabel("CDF")
    ax.set_title(
        r"IndivHausdorff Distances for $\varepsilon=" + str(epsilons[ideps]) + r"$"
    )


def plot_query_distortion(epsilons, ms, q, idq, r, idr, results, ax):
    eps_grid, m_grid = np.meshgrid(epsilons, ms)

    distortion_grid = np.array(
        [
            [results[i, j][idq][idr] for j in range(len(ms))]
            for i in range(len(epsilons))
        ]
    )

    ax.plot_surface(eps_grid, m_grid, distortion_grid, cmap="viridis", edgecolor="k")

    ax.set_xlabel("Epsilon")
    ax.set_ylabel("m")
    ax.set_zlabel("Distortion")
    ax.set_title(f"{q} Distortion vs Epsilon and m for $r={r}$")


def plot_indiv_hd_var_eps(epsilons, ms, idm, results, ax):
    step = 10
    for i, eps in enumerate(epsilons):
        indiv_hd = results[i, idm][1]

        counts = np.histogram(indiv_hd, bins=step)[0]
        cdf = np.cumsum(counts) / len(indiv_hd)
        ax.plot(np.arange(step), cdf, label=r"$\varepsilon=" + str(eps) + r"$")
        ax.legend()
    ax.set_xlabel("Indiv Hausdorff Distance")
    ax.set_ylabel("CDF")
    ax.set_title(r"IndivHausdorff Distances for $m=" + str(ms[idm]) + r"$")


if __name__ == "__main__":
    t_drive = pd.read_csv(
        "t-drive-trajectories/release/taxi_log_2008_by_id/cleaned_normalized.csv"
    )
    sample = t_drive[t_drive["date"] == "2008-02-03"]
    D = csv_db_to_numpy(sample)

    phi = 20
    ms = [40, 60, 80, 100]
    epsilons = [0.6, 0.8, 1.2, 1.6]

    results = Parallel(n_jobs=-1)(
        delayed(run_experiment)(D, eps, m, phi, i, j)
        for i, eps in enumerate(epsilons)
        for j, m in enumerate(ms)
    )
    results_idx = np.empty((len(epsilons), len(ms)), dtype=object)

    for i, j, hd, indiv_hd, psi_distortions, dai_distortions in results:  # type: ignore
        results_idx[i, j] = (hd, indiv_hd, psi_distortions, dai_distortions)

    plot_results(epsilons, ms, results_idx)
