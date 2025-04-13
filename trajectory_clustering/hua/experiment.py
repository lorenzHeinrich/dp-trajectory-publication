from matplotlib import gridspec, pyplot as plt

import numpy as np
import pandas as pd

from scipy.spatial.distance import directed_hausdorff

from trajectory_clustering.data.read_db import csv_db_to_numpy
from trajectory_clustering.hua.hua import Hua
from joblib import Parallel, delayed


def possibly_sometimes_inside(D, R, t_interval, uncertainty):
    """
    Calculate the number of trajectories that are possibly inside the circle at least once,
    considering a fixed uncertainty radius for each trajectory point.

    :param D: Original dataset
    :param R: Circle defined by center (x, y) and radius r
    :param t_interval: Time interval (tb, te)
    :param uncertainty: Fixed uncertainty radius for the entire trajectory, for each point
    :return: Number of trajectories that are possibly inside the circle at least once
    """
    tb, te = t_interval
    (x, y), r = R
    D_int = D.reshape(D.shape[0], -1, 2)[:, tb : te + 1, :]

    distances = np.sqrt((D_int[:, :, 0] - x) ** 2 + (D_int[:, :, 1] - y) ** 2)

    # Check if the distance from each point to the center of the circle is
    # less than or equal to the radius plus the uncertainty for any point
    inside = np.any(distances <= (r + uncertainty), axis=1)

    return np.sum(inside)


def definitely_always_inside(D, R, t_interval, uncertainty):
    """
    Calculate the number of trajectories that are definitely always inside the circle,
    considering the full uncertainty radius around each trajectory point.

    :param D: Original dataset
    :param R: Circle defined by center (x, y) and radius r
    :param t_interval: Time interval (tb, te)
    :param uncertainty: Fixed uncertainty radius for each trajectory point
    :return: Number of trajectories that are definitely always inside the circle
    """
    tb, te = t_interval
    (x, y), r = R

    D_int = D.reshape(D.shape[0], -1, 2)[:, tb : te + 1, :]

    distances = np.sqrt((D_int[:, :, 0] - x) ** 2 + (D_int[:, :, 1] - y) ** 2)

    # Check if the distance from each point to the center of the circle is
    # less than or equal to the radius minus the uncertainty for all points
    inside = np.all(distances <= (r - uncertainty), axis=1)

    return np.sum(inside)


def range_query_distortion(D, D_pub, Q):
    """
    Compute the range query distortion between the original and published datasets.
    :param D: Original dataset
    :param D_pub: Published dataset
    :param Q: Query function
    :return: Range query distortion
    """

    R_D = Q(D)
    R_D_pub = Q(D_pub)
    distortion = np.abs(R_D - R_D_pub) / (
        max(R_D, R_D_pub) + 1e-12  # Avoid division by zero
    )
    return distortion


def hausdorff(D, D_pub):
    """
    Compute the Hausdorff distance between two datasets.
    :param D: Original dataset
    :param D_pub: Published dataset
    :return: Hausdorff distance
    """
    return max(
        directed_hausdorff(D, D_pub)[0],
        directed_hausdorff(D_pub, D)[0],
    )


def indiv_hausdorff(D, D_pub):
    """
    Compute the "individual" Hausdorff distance for each T in D_pub
    .. math::
        IdvHD(T) = min_{T' in D} ||T - T'||_2
    """
    return np.array([np.min(np.linalg.norm(D - T, axis=1)) for T in D_pub])


def indiv_hausdorff_cdf(D, D_pub, bin_edges):
    """
    CDF of the invididual Hausdorff distance
    """
    idiv_hd = indiv_hausdorff(D, D_pub)
    counts = np.histogram(idiv_hd, bins=bin_edges)[0]

    cdf = np.cumsum(counts) / len(idiv_hd)
    return bin_edges[1:], cdf


def run_experiment(eps, m, phi, i, j):
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
        delayed(run_experiment)(eps, m, phi, i, j)
        for i, eps in enumerate(epsilons)
        for j, m in enumerate(ms)
    )
    results_idx = np.empty((len(epsilons), len(ms)), dtype=object)

    for i, j, hd, indiv_hd, psi_distortions, dai_distortions in results:  # type: ignore
        results_idx[i, j] = (hd, indiv_hd, psi_distortions, dai_distortions)

    plot_results(epsilons, ms, results_idx)
