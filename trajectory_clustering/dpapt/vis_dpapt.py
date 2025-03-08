from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
import sklearn.datasets

from trajectory_clustering.dpapt.dpapt import adaptive_noisy_grid, dpapt


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


def vis_data(D):
    x_bounds = (np.min(D[:, 0]) - 0.1, np.max(D[:, 0] + 0.1))
    y_bounds = (np.min(D[:, 1]) - 0.1, np.max(D[:, 1] + 0.1))

    eps = 2

    grid2 = adaptive_noisy_grid(D, (x_bounds, y_bounds), eps, True, alpha=0.5)

    _, ax = plt.subplots(figsize=(10, 10))
    visualize_grid(D, (x_bounds, y_bounds), grid2, ax)
    plt.show()


def moons():
    D = np.array(sklearn.datasets.make_moons(n_samples=1000, noise=0.1)[0])
    vis_data(D)


def blobs():
    D = np.array(sklearn.datasets.make_blobs(n_samples=1000, centers=3)[0])
    vis_data(D)


def vis_D(D, ax):
    for tr in D:
        ax.plot(tr[:, 0], tr[:, 1], "o-")


def vis_D_cells(traj_san, ax):
    for traj in traj_san:
        for cell in traj:
            ((x_l, x_u), (y_l, y_u)) = cell
            ax.add_patch(
                Rectangle((x_l, y_l), x_u - x_l, y_u - y_l, fill=False, linestyle="--")
            )
        centers = np.array(
            [((x_l + x_u) / 2, (y_l + y_u) / 2) for (x_l, x_u), (y_l, y_u) in traj]
        )
        ax.plot(centers[:, 0], centers[:, 1], "-")


# traj_len = 10
# D = np.array(
#     [
#         [
#             [np.random.uniform(0, 100), np.random.uniform(0, 100)]
#             for _ in range(traj_len)
#         ]
#         for _ in range(500)
#     ]
# )
# x_range = (0, 100)
# y_range = (0, 100)

# fig, ax = plt.subplots(1, 2, figsize=(10, 10))
# for a in ax:
#     a.set_aspect("equal", "box")
# vis_D(D, ax[0])

# traj_san, counts = dpapt(
#     D, (0, traj_len - 1), (x_range, y_range), eps=4, alpha=0.5, randomize=False
# )

# vis_D_cells(traj_san, ax[1])
# plt.show()
