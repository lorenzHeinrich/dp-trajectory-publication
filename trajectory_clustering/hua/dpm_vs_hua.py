from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import sklearn.cluster
import sklearn.metrics

from trajectory_clustering.hua.cluster import kmeans_partitioning
from trajectory_clustering.hua import dp_location_generalization
from trajectory_clustering.hua_with_dpm import dpm_location_generalization
from trajectory_clustering.trajectory import Trajectory, TrajectoryDatabase
from trajectory_clustering.hua.visualize import stepwise_plot


df = pd.read_csv(
    "t-drive-trajectories/release/taxi_log_2008_by_id/cleaned_normalized.csv"
)
sample = df[df["date"] == "2008-02-03"]
db = TrajectoryDatabase.from_dataframe(sample)
data = np.array([traj.as_array() for traj in db.trajectories.values()])


def kmeans_analysis():
    k_values = range(2, 100)
    silhouette_scores = []
    inertia = []
    for i in k_values:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        silhouette_scores.append(sklearn.metrics.silhouette_score(data, kmeans.labels_))
        inertia.append(kmeans.inertia_)

    # Create a dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Inertia plot (left y-axis)
    ax1.plot(k_values, inertia, "b-", label="Inertia")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Silhouette Score plot (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(k_values, silhouette_scores, "r-", label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # Titles and legends
    plt.title("Inertia and Silhouette Score vs. Number of Clusters")
    fig.tight_layout()
    plt.show()


def parse_partition(partition):
    centers = partition.mean_trajectories
    clusters = {
        label: [traj for traj in trajs.values()]
        for label, trajs in partition.labled_trajectories.items()
    }
    return centers, clusters


k_means_partition = kmeans_partitioning(db, m=14)
centers, clusters = parse_partition(k_means_partition)
stepwise_plot(centers.keys(), centers, clusters, bounds=(0, 100), draw_line=True)  # type: ignore

# hua_partition = dp_location_generalization(db, m=20, eps=1.0, phi=20)

# stepwise_plot(centers.keys(), centers, clusters, bounds=(0, 100))  # type: ignore

# dpm_partition = dpm_location_generalization(db, eps=2.0, delta=0.1, bounds=(0, 100))
# data = [traj.as_array() for traj in db.trajectories.values()]
# centers = {i: center for i, (center, _) in enumerate(dpm_partition)}
# clusters = {
#     i: [Trajectory.from_array(data[i], db.timestamps) for i in cluster]
#     for i, (_, cluster) in enumerate(dpm_partition)
# }
# stepwise_plot(centers.keys(), centers, clusters, bounds=(0, 100))
