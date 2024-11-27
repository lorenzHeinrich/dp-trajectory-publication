from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


# data: 'id', 'timestamp', 'longitude', 'latitude'
def animate_trajectories(data, interval=200, draw_line=False):
    # Initialize the plot
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    ax.set_title("Animated Trajectories")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Set axis limits based on the data
    ax.set_xlim(data["longitude"].min() - 1, data["longitude"].max() + 1)
    ax.set_ylim(data["latitude"].min() - 1, data["latitude"].max() + 1)

    # Create a dictionary of line objects for each trajectory
    lines = {}
    for key in data["id"].unique():
        (lines[key],) = ax.plot([], [], marker="o", label=f"Trajectory {key}")

    ax.legend()

    # Animation function
    def update(frame):
        """Update lines for the current timestamp."""
        current_data = data[data["timestamp"] == frame]
        for key, line in lines.items():
            traj_data = current_data[current_data["id"] == key]
            line.set_data(traj_data["longitude"], traj_data["latitude"])
        return lines.values()

    # Create animation
    timestamps = sorted(data["timestamp"].unique())
    return plt, FuncAnimation(fig, update, frames=timestamps, blit=True, interval=interval)
