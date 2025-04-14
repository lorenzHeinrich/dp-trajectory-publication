from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from trajectory_clustering.data.read_db import csv_db_to_numpy
from trajectory_clustering.dpapt.dpapt import (
    DPAPT,
    post_process_centroid,
    post_process_uniform,
)
from trajectory_clustering.hua.hua import Hua
from trajectory_clustering.metrics import (
    hausdorff,
    possibly_sometimes_inside,
    range_query_distortion,
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


def dpapt_vs(D, D_var, vars, var_name, dpapt_m, vs_m, measure):
    """
    Compare DPAPT with another method (e.g., VS) using a specific measure and varying parameter.

    :param D: Original dataset
    :param vars: List of values for the parameter to vary
    :param var_name: Name of the parameter to vary
    :param epsilons: List of epsilon values to test
    :param dpapt_m: the initialized DPAPT mechanism
    :param vs_m: the initialized mechanism to compare with
    :param measure: Measure to compare (e.g., "Hausdorff", "Distortion")
    :return: Comparison results
    """
    results = []
    for var in vars:

        dpapt_cells, dpapt_counts = dpapt_m(D, var)
        dpapt_trajects = post_process_uniform(dpapt_cells)
        dpapt_expanded = expand(dpapt_trajects, dpapt_counts)

        vs_result, vs_counts = vs_m(D_var(var), var)
        vs_expanded = expand(vs_result, vs_counts)

        dpapt_utility = measure(D_var(var), dpapt_expanded)
        vs_utility = measure(D_var(var), vs_expanded)
        results.append(
            {
                var_name: var,
                "dpapt_utility": dpapt_utility,
                "vs_utility": vs_utility,
            }
        )
    return results


def dpapt_vs_hua_var_eps(D, bounds, t_int, epsilons, measure):
    D_trunc = D[:, t_int[0] : t_int[1] + 1]
    hua = lambda D, eps: Hua(m=60, phi=100, eps=eps).publish(D)
    dpapt = lambda D, eps: DPAPT(alpha=0.5, beta=0.5, gamma=0.1, c1=10).publish(
        D, t_int, bounds, eps
    )

    return dpapt_vs(D, lambda _: D_trunc, epsilons, "eps", dpapt, hua, measure)


def dpapt_vs_hua_var_t_int(D, bounds, t_ints, eps, measure):

    hua = lambda D, _: Hua(m=60, phi=100, eps=eps).publish(D)
    dpapt = lambda D, t_int: DPAPT(alpha=0.5, beta=0.5, gamma=0.1, c1=10).publish(
        D, t_int, bounds, eps
    )
    D_var = lambda t_int: D[:, t_int[0] : t_int[1] + 1, :]
    return dpapt_vs(D, D_var, t_ints, "t_int", dpapt, hua, measure)


def hausdorff_vs_eps(D, bounds, t_int, epsilons):
    """
    Compare Hausdorff distances between DPAPT and another method (e.g., Hua).

    :param D: Original dataset
    :param bounds: Bounds for the dataset
    :param t_int: Time interval for the dataset
    :param epsilons: List of epsilon values to test
    :return: Comparison results
    """
    results = dpapt_vs_hua_var_eps(D, bounds, t_int, epsilons, hausdorff)
    plot_results(
        results,
        variable="eps",
        x_label=r"$\varepsilon$",
        y_label="Hausdorff Distance",
        title="Hausdorff Distances of DPAPT vs. Hua",
    )


def query_distortions_vs_eps(D, bounds, t_int, epsilons):
    """
    Compare range query distortion between DPAPT and another method (e.g., Hua).

    :param D: Original dataset
    :param bounds: Bounds for the dataset
    :param t_int: Time interval for the dataset
    :param epsilons: List of epsilon values to test
    :return: Comparison results
    """

    def psi_distortion(D, D_pub):
        psi_distortions = []
        for _ in range(50):
            R = np.random.uniform(0, 100, 2)
            t_interval = np.random.randint(0, t_int[1] - t_int[0], 2)
            psi = lambda D: possibly_sometimes_inside(D, (R, 0.5), t_interval, 0.5)
            psi_distortion = range_query_distortion(D, D_pub, psi)
            psi_distortions.append(psi_distortion)
        return np.mean(psi_distortions)

    results = dpapt_vs_hua_var_eps(D, bounds, t_int, epsilons, psi_distortion)
    plot_results(
        results,
        variable="eps",
        x_label=r"$\varepsilon$",
        y_label="Range Query Distortion",
        title="Range Query Distortion of DPAPT vs. Hua avg. over 50 queries",
    )


def compare_hausdorff_vs_t_int(D, bounds, t_ints, eps):
    """
    Compare Hausdorff distances between DPAPT and another method (e.g., Hua) for varying time intervals.

    :param D: Original dataset
    :param bounds: Bounds for the dataset
    :param t_ints: List of time intervals to test
    :param eps: Epsilon value to use
    :return: Comparison results
    """
    results = dpapt_vs_hua_var_t_int(D, bounds, t_ints, eps, hausdorff)
    plot_results(
        results,
        variable="t_int",
        x_label="Time Interval",
        y_label="Hausdorff Distance",
        title="Hausdorff Distances of DPAPT vs. Hua",
    )


def plot_results(results, variable, x_label, y_label, title):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    dpapt_utility = [result["dpapt_utility"] for result in results]
    vs_utility = [result["vs_utility"] for result in results]
    eps = [result[variable] for result in results]
    ax.plot(eps, dpapt_utility, label="DPAPT", marker="x")
    ax.plot(eps, vs_utility, label="Hua", marker="o")
    ax.set_xlabel(x_label)
    ax.set_xticks(eps)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(
        "t-drive-trajectories/release/taxi_log_2008_by_id/cleaned_normalized.csv"
    )
    df = df[df["date"] == "2008-02-03"]
    D = csv_db_to_numpy(df)
    bounds = ((0, 100), (0, 100))
    t_int = (0, 2)
    epsilons = [0.8, 1.2, 1.6, 2.0]
    # compare_hausdorff_vs_hua(D, bounds, t_int, epsilons)
    # query_distortions_vs_eps(D, bounds, t_int, epsilons)
    t_ints = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    eps = 2
    compare_hausdorff_vs_t_int(D, bounds, t_ints, eps)
