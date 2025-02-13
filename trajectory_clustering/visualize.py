import itertools
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd

from trajectory_clustering.cluster import Partition, kmeans_partitioning
from trajectory_clustering.trajectory import TrajectoryDatabase


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


def stepwise_plot(partition: Partition):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    ax.set_title("Stepwise Trajectories")
    max_x, max_y = max(
        [
            (p.location.x, p.location.y)
            for group in partition.labled_trajectories.values()
            for traj in group.values()
            for p in traj.st_points.values()
        ]
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    colors = plt.cm.get_cmap("tab10", partition.n_labels)
    step = [0]

    def update(step_label):
        ax.clear()
        ax.set_title(f"Group {step_label}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(0, int(max_x) + 2)
        ax.set_ylim(0, int(max_y) + 2)

        group = partition.labled_trajectories[step_label]
        for traj in group.values():
            x = [p.location.x for p in traj.st_points.values()]
            y = [p.location.y for p in traj.st_points.values()]
            timestamps = [p.timestamp for p in traj.st_points.values()]
            ax.plot(x, y, color=colors(step_label))
            ax.scatter(x, y, color=colors(step_label))

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

            x = [
                p.location.x
                for p in partition.mean_trajectories[step_label].st_points.values()
            ]
            y = [
                p.location.y
                for p in partition.mean_trajectories[step_label].st_points.values()
            ]
            ax.plot(x, y, color="black")
            ax.scatter(x, y, color="black")

            fig.canvas.draw_idle()

    def next_step(event):
        step[0] = (step[0] + 1) % partition.n_labels
        update(step[0])

    update(step[0])

    ax_button = plt.axes((0.8, 0.01, 0.1, 0.075))
    button = Button(ax_button, "Next")
    button.on_clicked(next_step)

    plt.show()


def main():
    data = pd.read_csv("tests/data/fake-trajectories_4x5.csv")
    db = TrajectoryDatabase.from_dataframe(data)
    partition = kmeans_partitioning(db, 2)
    stepwise_plot(partition)


if __name__ == "__main__":
    main()
