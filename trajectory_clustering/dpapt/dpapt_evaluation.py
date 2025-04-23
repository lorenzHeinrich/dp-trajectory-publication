from time import time
import folium
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import seaborn as sns

from trajectory_clustering.data.read_db import t_drive
from trajectory_clustering.dpapt import dpapt
from trajectory_clustering.dpapt.adaptive_cells import AdaptiveCells
from trajectory_clustering.dpapt.dpapt import DPAPT, post_process_centroid


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


def visualize_grid(data, bounds, L2Grids, ax: Axes):
    ax.set_aspect("equal", "box")
    ax.scatter(data[:, 0], data[:, 1])

    m1 = int(np.sqrt(len(L2Grids)))
    ((xl, xu), (yl, yu)) = bounds
    xl1_step = (xu - xl) / m1
    yl1_step = (yu - yl) / m1

    ax.set_xlim(xl, xu)
    ax.set_ylim(yl, yu)
    hlines = np.linspace(xl, xu, m1, endpoint=False)
    vlines = np.linspace(yl, yu, m1, endpoint=False)
    for i in range(m1):
        ax.axvline(hlines[i], c="gray", linestyle="--", lw=2.0)
        ax.axhline(vlines[i], c="gray", linestyle="--", lw=2.0)

    for i, j in np.ndindex((m1, m1)):
        m2, _, _, _, l2Grid = L2Grids[(i, j)]
        xl2_step = xl1_step / m2
        yl2_step = yl1_step / m2
        cell_xl = xl + i * xl1_step
        cell_yl = yl + j * yl1_step
        cell_hlines = np.linspace(cell_xl, cell_xl + xl1_step, m2, endpoint=False)
        cell_vlines = np.linspace(cell_yl, cell_yl + yl1_step, m2, endpoint=False)
        for x in range(m2):
            ax.hlines(cell_vlines[x], cell_xl, cell_xl + xl1_step, color="gray", lw=0.5)
            ax.vlines(cell_hlines[x], cell_yl, cell_yl + yl1_step, color="gray", lw=0.5)

        for x, y in np.ndindex(l2Grid.shape):
            ax.text(
                cell_xl + x * xl2_step + xl2_step / 2,
                cell_yl + y * yl2_step + yl2_step / 2,
                f"{l2Grid[x, y]:.2f}",
                ha="center",
                va="center",
            )


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


def sanity_check():
    D = np.array(
        [[[4, 4], [6, 4], [8, 4]], [[2, 2], [4, 2], [6, 2]], [[3, 1], [5, 1], [7, 1]]]
    )
    t_interval = (0, 1)
    bounds = ((0, 10), (0, 10))
    eps = 1.0

    dpapt = DPAPT(
        randomize=False,
        ac=AdaptiveCells(randomize=False, n_clusters=None, do_filter=True),
    )
    D_areas, counts = dpapt.publish(D, t_interval, bounds, eps)
    print(D_areas, counts)
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    vis_D_cells(D_areas, ax=ax)
    vis_D(D, ax=ax)
    plt.show()


def vis_D(D, ax):
    for tr in D:
        ax.plot(tr[:, 0], tr[:, 1], "o-")


def vis_D_cells(traj_san, ax):
    pallate = sns.color_palette("colorblind", n_colors=len(traj_san))
    for i, traj in enumerate(traj_san):
        for area in traj:
            for cell in area.cells:
                ((x_l, x_u), (y_l, y_u)) = cell
                ax.add_patch(
                    Rectangle(
                        (x_l, y_l),
                        x_u - x_l,
                        y_u - y_l,
                        fill=True,
                        alpha=0.5,
                        color=pallate[i],
                    )
                )
        centers = np.array([area.center for area in traj])
        ax.plot(centers[:, 0], centers[:, 1], "-")

        sns.scatterplot(
            x=centers[:, 0],
            y=centers[:, 1],
            ax=ax,
            color="red",
            marker="x",
        )


def vis_D_folium(D, map_obj, data_bounds):
    for tr in D:
        coords = [(lat, lon) for lon, lat in tr]
        coords = [denormalise_coords(lon, lat, data_bounds) for lon, lat in coords]
        folium.PolyLine(coords, color="blue", weight=2).add_to(map_obj)
        for lon, lat in coords:
            folium.Marker(
                [lat, lon], icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(map_obj)


def vis_D_cells_folium(D_areas, map_obj, data_bounds):
    palette = sns.color_palette("colorblind", n_colors=len(D_areas))
    for i, traj in enumerate(D_areas):
        for area in traj:
            for (xl, xu), (yl, yu) in area.cells:
                # Denormalise corner coordinates
                lon_l, lat_l = denormalise_coords(xl, yl, data_bounds)
                lon_u, lat_u = denormalise_coords(xu, yu, data_bounds)
                bounds = [(lat_l, lon_l), (lat_u, lon_u)]  # Folium expects (lat, lon)

                folium.Rectangle(
                    bounds=bounds,
                    color=matplotlib.colors.to_hex(palette[i]),
                    fill=True,
                    fill_opacity=0.3,
                    weight=1,
                ).add_to(map_obj)

            # Optional: add cluster centroid as a marker
            lon, lat = denormalise_coords(area.center[0], area.center[1], data_bounds)
            folium.CircleMarker([lat, lon], radius=3, color="black").add_to(map_obj)


def denormalise_coords(norm_lon, norm_lat, bounds):
    """
    norm_lon, norm_lat: float or array-like values in [0, 100]
    bounds: ((min_lon, max_lon), (min_lat, max_lat))
    Returns: (lon, lat) in original coordinate space
    """
    (min_lon, max_lon), (min_lat, max_lat) = bounds
    lon = norm_lon / 100 * (max_lon - min_lon) + min_lon
    lat = norm_lat / 100 * (max_lat - min_lat) + min_lat
    return lon, lat


if __name__ == "__main__":
    # min_long: 116.03725, max_long: 116.85369
    # min_lat: 39.62473, max_lat: 40.27222
    # scaling factor: 100
    min_long, max_long = 116.03725, 116.85369
    min_lat, max_lat = 39.62473, 40.27222
    data_bounds = ((min_long, max_long), (min_lat, max_lat))

    D, bounds = t_drive("small")
    # D = D[:100]
    ac = AdaptiveCells(n_clusters=40, do_filter=False)
    eps = 5.0
    t_int = (0, 0)
    D_areas, counts = DPAPT(ac=ac).publish(D, t_int, bounds, eps)

    center_lon, center_lat = denormalise_coords(
        np.mean(D[:, :, 0]), np.mean(D[:, :, 1]), data_bounds
    )
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)  # type: ignore

    vis_D_folium(D[:, :1], m, data_bounds)
    vis_D_cells_folium(D_areas, m, data_bounds)

    m.save("dpapt_map.html")  # Save to view in browser
