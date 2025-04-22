from time import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff

from trajectory_clustering.dpapt.dpapt import DPAPT, post_process_centroid
from trajectory_clustering.dpapt.vis_dpapt import vis_D, vis_D_cells


def normalized_hausdorff(D1, D2):
    return max(
        directed_hausdorff(D1.reshape(D1.shape[0], -1), D2.reshape(D2.shape[0], -1))[0],
        directed_hausdorff(D2.reshape(D2.shape[0], -1), D1.reshape(D1.shape[0], -1))[0],
    ) / np.sqrt(D1.shape[1])


thresh_std_var = lambda eps: 2 * np.sqrt(2) / eps
thresh_var = lambda eps: 4 / eps**2


def dpapt_experiment(
    D,
    t_interval,
    eps,
    alpha,
    c1=10,
    thresh_grid=thresh_std_var,
    thresh_traj=thresh_std_var,
    randomize=True,
    measures=[normalized_hausdorff],
):

    x_range = (np.min(D[:, :, 0]) - 0.1, np.max(D[:, :, 0] + 0.1))
    y_range = (np.min(D[:, :, 1]) - 0.1, np.max(D[:, :, 1] + 0.1))

    dpapt_ = DPAPT(
        alpha=alpha,
        beta=alpha,
        gamma=0.1,
        c=c1,
        thresh_grid=thresh_grid,
        thresh_traj=thresh_traj,
        randomize=randomize,
    )

    D_cell, _ = dpapt_.publish(D, t_interval, (x_range, y_range), eps)
    D_post = post_process_centroid(D_cell)

    D_trimmed = D[:, t_interval[0] : t_interval[1] + 1]

    results = [m(D_trimmed, D_post) for m in measures]

    print(f"Experiment with ε={eps}, α={alpha}, c1={c1}: {results}")

    return results


def plot_results(x, y, title, x_label, y_label, label, marker, colors, line, ax):

    for i in range(len(y)):
        ax.plot(
            x,
            y[i],
            marker=marker,
            linestyle=line,
            label=f"{label[i]}",
            color=colors(i),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def time_intervals(
    D, min_t, max_t, epsilons=[2, 3, 4], alpha=0.5, c1=10, ax=None, eps_colors=None
):
    time_intervals = list(range(min_t, max_t + 1))

    plt.figure(figsize=(8, 5)) if ax is None else plt.sca(ax)
    eps_colors = (
        plt.cm.get_cmap("tab10", len(epsilons)) if eps_colors is None else eps_colors
    )

    results = np.ndarray((len(epsilons), len(time_intervals)))

    for i, eps in enumerate(epsilons):

        for j, t in enumerate(time_intervals):
            t_interval = (0, t)

            result = dpapt_experiment(
                D,
                t_interval,
                eps,
                alpha,
                c1,
                thresh_grid=thresh_std_var,
                thresh_traj=thresh_std_var,
                measures=[normalized_hausdorff],
            )

            results[i, j] = result[0]

    plot_results(
        time_intervals,
        results,
        "Hausdorff Distance vs. Time Interval Length for Different Epsilon Values",
        "Time Interval Length",
        "Hausdorff Distance",
        [f"ε={eps}" for eps in epsilons],
        "o",
        eps_colors,
        "--",
        ax,
    )


def evaluate_c1(D, t_interval, epsilons, alpha, c1_values, ax=None, c1_colors=None):
    plt.figure(figsize=(8, 5)) if ax is None else plt.sca(ax)
    c1_colors = (
        plt.cm.get_cmap("tab10", len(c1_values)) if c1_colors is None else c1_colors
    )

    results = np.ndarray((len(epsilons), len(c1_values)))

    for i, eps in enumerate(epsilons):
        for j, c1 in enumerate(c1_values):
            result = dpapt_experiment(
                D, t_interval, eps, alpha, c1, [normalized_hausdorff]
            )
            results[i, j] = result[0]

    plot_results(
        c1_values,
        results,
        "Hausdorff Distance vs. c1 for Fixed Time Interval and Different Epsilon Values",
        "c1",
        "Hausdorff Distance",
        [f"ε={eps}" for eps in epsilons],
        "o",
        c1_colors,
        "--",
        ax,
    )


def eval_taxis():
    D = D_taxis(7, 700)
    _, axs = plt.subplots(1, 1)
    time_intervals(D, 0, 4, epsilons=[1, 2, 3, 4], alpha=0.5, c1=10, ax=axs)
    plt.show()


def eval_gowalla():
    m = 7
    n = 10000
    D = D_gowalla(m, n)
    _, axs = plt.subplots(1, 1)
    time_intervals(D, 0, 2, epsilons=[1, 2, 3, 4], alpha=0.5, c1=10, ax=axs)
    plt.show()


def sanity_check(n, m):
    ((x_l, x_u), (y_l, y_u)) = ((0, 10), (0, 10))
    D = D_random(n, m, (x_l, x_u), (y_l, y_u))

    bounds = ((x_l, x_u), (y_l, y_u))
    t_interval = (0, m - 1)
    dpapt_ = DPAPT(
        alpha=0.5,
        beta=0.5,
        gamma=0.1,
        c=2,
        thresh_grid=thresh_std_var,
        thresh_traj=thresh_std_var,
        randomize=False,
    )
    D_cell, _ = dpapt_.publish(D, t_interval, bounds, eps=1)
    _, axs = plt.subplots(1, 2)
    vis_D(D, axs[0])
    vis_D_cells(D_cell, axs[1])
    plt.show()


def D_random(n, m, x_range, y_range):
    ((x_l, x_u), (y_l, y_u)) = x_range, y_range
    D = np.array(
        [
            [
                [np.random.uniform(x_l, x_u), np.random.uniform(y_l, y_u)]
                for _ in range(m)
            ]
            for _ in range(n)
        ]
    )
    return D


def D_gowalla(m, n):
    df = pd.read_csv(f"gowalla/merged_trajectories_length_{m}.csv", nrows=m * n)
    D = np.ndarray((n, m, 2))
    print(f"Importing gowalla dataset with {n} trajectories of length {m}:")
    for idx, id in enumerate(df["traj_id"].unique()):
        D[idx] = df[df["traj_id"] == id][["latitude", "longitude"]].values
        if idx % (n // 10) == 0:
            print(f"{idx/n*100:.0f}%", end="\r")
    return D


def D_taxis(m, n):
    df = pd.read_csv(
        "t-drive-trajectories/release/taxi_log_2008_by_id/cleaned_normalized.csv"
    )
    sample = df[df["date"] == "2008-02-03"]

    D = np.ndarray((0, 37, 2))
    for id in sample["id"].unique():
        D = np.concatenate(
            (D, [sample[sample["id"] == id][["latitude", "longitude"]].values])
        )
    return D[:n, :m]


def benchmark_size(D, t_interval, epsilons=[1, 2, 3], c1=10, randomize=True, axs=None):
    sizes = np.arange(len(D) // 5, len(D), len(D) // 5)
    times = np.zeros((len(epsilons), len(sizes)))
    for i, eps in enumerate(epsilons):
        for j, n in enumerate(sizes):
            D_subset = D[:n]
            start = time()
            dpapt_experiment(
                D_subset,
                t_interval,
                eps,
                0.5,
                c1,
                thresh_traj=thresh_var,
                thresh_grid=thresh_var,
                randomize=randomize,
                measures=[],
            )
            times[i, j] = time()
            print(f"Benchmark with ε={eps}, n={n}: {times[i, j] - start:.2f} s")
    plot_results(
        sizes,
        times,
        "Execution Time vs. Dataset Size",
        "Dataset Size",
        "Execution Time (s)",
        [f"ε={eps}" for eps in epsilons],
        "o",
        plt.cm.get_cmap("tab10", 1),
        "--",
        axs,
    )


def benchmark_interval(D, epsilons=[0.5, 1, 2], c1=10, axs=None):
    intervals = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    times = np.zeros((len(epsilons), len(intervals)))
    for i, eps in enumerate(epsilons):
        for j, t in enumerate(intervals):
            start = time()
            dpapt_experiment(
                D,
                t,
                eps,
                0.5,
                c1,
                thresh_traj=thresh_var,
                thresh_grid=thresh_var,
                measures=[],
            )
            times[i, j] = time() - start
            print(f"Benchmark with ε={eps}, t={t}: {times[i, j] - start:.2f} s")
    plot_results(
        intervals,
        times,
        "Execution Time vs. Time Interval",
        "Time Interval",
        "Execution Time (s)",
        [f"ε={eps}" for eps in epsilons],
        "o",
        plt.cm.get_cmap("tab10", 1),
        "--",
        axs,
    )


if __name__ == "__main__":
    # eval_taxis()
    # eval_gowalla()
    # sanity_check(10, 10)
    D = D_gowalla(7, 20000)
    _, axs = plt.subplots(1, 1)
    benchmark_size(D, (0, 4), randomize=True, axs=axs)
    benchmark_interval(D, axs=axs)
    plt.show()
