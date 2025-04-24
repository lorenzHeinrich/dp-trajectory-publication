import folium
import matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

from trajectory_clustering.data.read_db import t_drive
from trajectory_clustering.data.transform import xy_to_lonlat
from trajectory_clustering.dpapt.adaptive_cells import AdaptiveCells
from trajectory_clustering.dpapt.dpapt import DPAPT


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


def vis_D_folium(D, map_obj):
    for tr in D:
        coords = [(lat, lon) for lon, lat in tr]
        coords = [xy_to_lonlat(lon, lat, "Beijing") for lon, lat in coords]
        folium.PolyLine(coords, color="blue", weight=2).add_to(map_obj)
        for lon, lat in coords:
            folium.Marker(
                [lat, lon], icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(map_obj)


def vis_D_cells_folium(D_areas, map_obj):
    palette = sns.color_palette("colorblind", n_colors=len(D_areas))
    for i, traj in enumerate(D_areas):
        for area in traj:
            for (xl, xu), (yl, yu) in area.cells:
                # Denormalise corner coordinates
                lon_l, lat_l = xy_to_lonlat(xl, yl, "Beijing")
                lon_u, lat_u = xy_to_lonlat(xu, yu, "Beijing")
                bounds = [(lat_l, lon_l), (lat_u, lon_u)]  # Folium expects (lat, lon)

                folium.Rectangle(
                    bounds=bounds,
                    color=matplotlib.colors.to_hex(palette[i]),
                    fill=True,
                    fill_opacity=0.3,
                    weight=1,
                ).add_to(map_obj)

            # Optional: add cluster centroid as a marker
            lon, lat = xy_to_lonlat(area.center[0], area.center[1], "Beijing")
            folium.CircleMarker([lat, lon], radius=3, color="black").add_to(map_obj)


if __name__ == "__main__":
    D, bounds = t_drive("small")
    ac = AdaptiveCells(n_clusters=20, do_filter=False)
    eps = 5.0
    t_int = (0, 4)
    D_areas, counts = DPAPT(ac=ac).publish(D, t_int, bounds, eps)

    center_lon, center_lat = xy_to_lonlat(
        float(np.mean(D[:, :, 0])), float(np.mean(D[:, :, 1])), "Beijing"
    )
    m = folium.Map(location=[center_lat, center_lon], tiles="OpenStreetMap.DE", zoom_start=10)  # type: ignore

    # vis_D_folium(D[:, :2], m)
    vis_D_cells_folium(D_areas[4:5], m)

    m.save("maps/dpapt_map.html")  # Save to view in browser
