from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from trajectory_clustering.dpapt import dpapt
from trajectory_clustering.dpapt.vis_dpapt import vis_D, vis_D_cells


df = pd.read_csv(
    "t-drive-trajectories/release/taxi_log_2008_by_id/cleaned_normalized.csv"
)
sample = df[df["date"] == "2008-02-03"]

D = np.ndarray((0, 37, 2))
for id in sample["id"].unique():
    D = np.concatenate(
        (D, [sample[sample["id"] == id][["latitude", "longitude"]].values])
    )


x_range = (np.min(D[:, :, 0]) - 0.1, np.max(D[:, :, 0] + 0.1))
y_range = (np.min(D[:, :, 1]) - 0.1, np.max(D[:, :, 1] + 0.1))
D_san, counts = dpapt.dpapt(
    D,
    t_interval=(0, 7),
    bounds=(x_range, y_range),
    eps=14,
    alpha=0.5,
    randomize=True,
    c1=10,
)
print(counts)
fig, ax = plt.subplots(1, 2)
vis_D(D, ax[0])
vis_D_cells(D_san, ax[1])
plt.show()
