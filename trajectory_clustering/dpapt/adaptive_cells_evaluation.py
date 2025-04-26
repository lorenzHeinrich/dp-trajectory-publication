import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

from trajectory_clustering.data.read_db import t_drive
from trajectory_clustering.dpapt.adaptive_cells import AdaptiveCells


def draw_cells(cells, ax, labels=None, centers=None):
    palette = (
        sns.color_palette("colorblind", n_colors=len(np.unique(labels)))
        if labels is not None
        else sns.color_palette("colorblind", n_colors=1)
    )
    for cell, label in zip(
        cells, np.zeros((len(cells)), dtype=int) if labels is None else labels
    ):
        (xl, xu), (yl, yu) = cell
        rect = Rectangle(
            (xl, yl),
            xu - xl,
            yu - yl,
            fill=True,
            alpha=0.8,
            color=palette[label],
            # linewidth=1,
        )
        ax.add_patch(rect)
    # scatter centers
    if centers is not None:
        sns.scatterplot(
            x=centers[:, 0],
            y=centers[:, 1],
            ax=ax,
            color="red",
            marker="x",
            label="Cluster Centers",
        )


def compare_default_var_eps(L, bounds, epsilons=[0.5, 2], out_dir="./"):
    _, axs = plt.subplots(1, len(epsilons))
    for ax, eps in zip(axs, epsilons):
        areas = AdaptiveCells().adaptive_cells(L, bounds, eps)  # type: ignore
        cells = [cell for area in areas for cell in area.cells]
        draw_cells(cells, ax)
        ax.set_xlim(x_lb, x_ub)
        ax.set_ylim(y_lb, y_ub)
        ax.set_title(f"ε={eps}, {len(cells)} unique cells")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        ax.grid()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/adaptive_cells_eps.pdf", bbox_inches="tight")


def compare_m2_eps(L, bounds, f_m1, f_m2, c=10, epsilons=[0.5, 2], out_dir="./"):
    _, axs = plt.subplots(1, len(epsilons))
    for ax, eps in zip(axs, epsilons):
        areas = AdaptiveCells(f_m1=f_m1, f_m2=f_m2, c=c).adaptive_cells(  # type: ignore
            L, bounds, eps
        )
        cells = [cell for area in areas for cell in area.cells]
        draw_cells(cells, ax)
        ax.set_xlim(x_lb, x_ub)
        ax.set_ylim(y_lb, y_ub)
        ax.set_title(f"ε={eps}, {len(cells)} unique cells")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        ax.grid()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/adaptive_cells_eps_fm_c{c}.pdf", bbox_inches="tight")


def compare_cluster(L, bounds, eps, do_filter, k=[10, 20]):
    _, axs = plt.subplots(1, len(k))
    for ax, k_ in zip(axs, k):
        areas = AdaptiveCells(n_clusters=k_, do_filter=do_filter).adaptive_cells(  # type: ignore
            L, bounds, eps
        )
        cells = [cell for area in areas for cell in area.cells]
        labels = [i for i, area in enumerate(areas) for _ in area.cells]
        centers = np.array([area.center for area in areas])
        draw_cells(cells, ax, labels, centers)
        ax.set_xlim(x_lb, x_ub)
        ax.set_ylim(y_lb, y_ub)
        ax.set_title(f"ε={eps}, {k_} clusters")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        ax.grid()
    plt.tight_layout()
    plt.savefig(
        f"figures/adaptive_cells/adaptive_cells_eps{eps}_k{"_filtered" if do_filter else ""}.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":

    D, ((x_lb, x_ub), (y_lb, y_ub)) = t_drive("small")
    bounds = ((x_lb, x_ub), (y_lb, y_ub))
    L = D[:, 0]

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    out_dir = "figures/adaptive_cells"
    os.makedirs(out_dir, exist_ok=True)

    compare_default_var_eps(L, bounds, epsilons=[0.5, 1.5, 2.5], out_dir=out_dir)

    f_m1 = lambda N, eps, c: max(5, 1 / 4 * np.ceil(np.ceil(np.sqrt(N / c))))
    f_m2 = lambda nc, eps, c: int(np.ceil(np.sqrt(max(1, nc) / (c / 2))))
    compare_m2_eps(
        L,
        bounds,
        f_m1=f_m1,
        f_m2=f_m2,
        epsilons=[0.5, 1.5, 2.5],
        out_dir=out_dir,
        c=200,
    )
    compare_m2_eps(
        L,
        bounds,
        f_m1=f_m1,
        f_m2=f_m2,
        epsilons=[0.5, 1.5, 2.5],
        out_dir=out_dir,
        c=200,
    )

    for do_filter in [True, False]:
        compare_cluster(L, bounds, eps=0.5, do_filter=do_filter, k=[15, 30])
        compare_cluster(L, bounds, eps=1.5, do_filter=do_filter, k=[15, 30])
        compare_cluster(L, bounds, eps=2.5, do_filter=do_filter, k=[15, 30])
