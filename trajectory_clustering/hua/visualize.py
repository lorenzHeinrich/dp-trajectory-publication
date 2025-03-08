import itertools
from turtle import up
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd

from trajectory_clustering.hua.cluster import Partition, kmeans_partitioning
from trajectory_clustering.base.trajectory import Trajectory, TrajectoryDatabase


# data: 'id', 'timestamp', 'longitude', 'latitude'
def animate_trajectories(data, interval=200, draw_line=False):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    ax.set_title("Animated Trajectories")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xlim(data["longitude"].min() - 1, data["longitude"].max() + 1)
    ax.set_ylim(data["latitude"].min() - 1, data["latitude"].max() + 1)

    lines = {}
    for key in data["id"].unique():
        (lines[key],) = ax.plot([], [], marker="o", label=f"Trajectory {key}")

    ax.legend()

    def update(frame):
        """Update lines for the current timestamp."""
        current_data = (
            data[data["timestamp"] <= frame]
            if draw_line
            else data[data["timestamp"] == frame]
        )
        for key, line in lines.items():
            traj_data = current_data[current_data["id"] == key]
            line.set_data(traj_data["longitude"], traj_data["latitude"])
        return lines.values()

    timestamps = sorted(data["timestamp"].unique())
    return plt, FuncAnimation(
        fig, update, frames=timestamps, blit=True, interval=interval
    )


def trajectory_to_dataframe(trajectory):
    return pd.DataFrame(
        [
            {
                "id": trajectory.id,
                "timestamp": p.timestamp,
                "longitude": p.location.x,
                "latitude": p.location.y,
            }
            for p in trajectory.st_points.values()
        ]
    )


def get_trajectory_figure():
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    ax.set_title("Trajectories")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig, ax


def plot_trajectories(traj, ax, color, label=None):
    data = trajectory_to_dataframe(traj)
    for key in data["id"].unique():
        traj_data = data[data["id"] == key]
        ax.plot(
            traj_data["longitude"],
            traj_data["latitude"],
            label=f"Trajectory {key if label == None else label}",
            color=color,
        )
    ax.legend()
    return ax


def stepwise_plot(
    labels,
    centers: dict[int, Trajectory],
    clusters: dict[int, list[Trajectory]],
    bounds: tuple[int, int],
    show_timestamps=False,
    draw_line=False,
    cluster_marker_size=5,
    label_marker_size=20,
):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    ax.set_title("Stepwise Trajectories")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    colors = plt.cm.get_cmap("tab10", len(labels))
    step = [0]

    def update(step_label):
        ax.clear()
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])
        ax.set_title(f"Group {step_label}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        cluster = clusters[step_label]
        for traj in cluster:
            x = [p.location.x for p in traj.st_points.values()]
            y = [p.location.y for p in traj.st_points.values()]
            timestamps = [p.timestamp for p in traj.st_points.values()]
            if draw_line:
                ax.plot(x, y, color=colors(step_label))
            ax.scatter(x, y, color=colors(step_label), s=cluster_marker_size)

            if show_timestamps:
                offsets = itertools.cycle([(0, 10), (10, 0), (0, -10), (-10, 0)])
                for i, txt in enumerate(timestamps):
                    offset = next(offsets)
                    ax.annotate(
                        str(txt),
                        (float(x[i]), float(y[i])),
                        textcoords="offset points",
                        xytext=offset,
                        ha="center",
                        fontsize=12,
                    )

        x = [p.location.x for p in centers[step_label].st_points.values()]
        y = [p.location.y for p in centers[step_label].st_points.values()]
        if draw_line:
            ax.plot(x, y, color="red")
        ax.scatter(x, y, color="red", marker="x", s=label_marker_size)

        fig.canvas.draw_idle()

    def next_step(event):
        step[0] = (step[0] + 1) % len(labels)
        update(step[0])

    def prev_step(event):
        step[0] = (step[0] - 1) % len(labels)
        update(step[0])

    update(step[0])

    # next
    next_button = plt.axes((0.8, 0.01, 0.1, 0.075))
    next_btn = Button(next_button, "Next")
    next_btn.on_clicked(next_step)

    # prev
    prev_button = plt.axes((0.7, 0.01, 0.1, 0.075))
    prev_btn = Button(prev_button, "Prev")
    prev_btn.on_clicked(prev_step)

    plt.show()
