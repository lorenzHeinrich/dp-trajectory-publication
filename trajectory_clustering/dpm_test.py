import numpy as np
import pandas as pd
from dpm.dpm import DPM
import matplotlib.pyplot as plt

from trajectory_clustering.hua_with_dpm import dpm_location_generalization
from trajectory_clustering.trajectory import TrajectoryDatabase
from trajectory_clustering.visualize import get_trajectory_figure, plot_trajectories

df = pd.read_csv("tests/data/fake-trajectories_500x5.csv")
traj_db = TrajectoryDatabase.from_dataframe(df)

generalized = dpm_location_generalization(100, 0.1, traj_db, (0, 10))
generalized = list(filter(lambda x: len(x[1]) > 0, generalized))
fig, ax = get_trajectory_figure()
colors = plt.cm.get_cmap("tab10", len(generalized))
for i, (traj, cluster) in enumerate(generalized):
    for k in cluster:
        plot_trajectories(traj_db.trajectories[k], ax, color=colors(i), label=i)
    plot_trajectories(traj, ax, color="red")
plt.show()
