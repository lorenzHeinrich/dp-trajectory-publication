from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from dpm.dpm import DPM

from trajectory_clustering.trajectory import Trajectory, TrajectoryDatabase

df = pd.read_csv("t-drive-trajectories/release/taxi_log_2008_by_id/cleaned.csv")
sample = df[df["date"] == "2008-02-03"]

lat_iqr = sample["latitude"].quantile(0.75) - sample["latitude"].quantile(0.25)
long_iqr = sample["longitude"].quantile(0.75) - sample["longitude"].quantile(0.25)
print(f"Latitude IQR: {lat_iqr} Longitude IQR: {long_iqr}")
lat_lwr = sample["latitude"].quantile(0.25) - 1.5 * lat_iqr
lat_upr = sample["latitude"].quantile(0.75) + 1.5 * lat_iqr
long_lwr = sample["longitude"].quantile(0.25) - 1.5 * long_iqr
long_upr = sample["longitude"].quantile(0.75) + 1.5 * long_iqr
print(
    f"Latitude outlier: {(lat_lwr, lat_upr)} Longitude outlier: {(long_lwr, long_upr)}"
)

# Filter out outliers
sample = sample[
    (sample["latitude"] >= lat_lwr)
    & (sample["latitude"] <= lat_upr)
    & (sample["longitude"] >= long_lwr)
    & (sample["longitude"] <= long_upr)
]

# remove trajectories with less then 37 points per date
points_per_day = sample.groupby(["id", "date"]).size().reset_index(name="counts")
sample = sample[sample["id"].isin(points_per_day[points_per_day["counts"] >= 37]["id"])]

min_long, max_long = sample["longitude"].min(), sample["longitude"].max()
min_lat, max_lat = sample["latitude"].min(), sample["latitude"].max()

# nomralize the data
sample["longitude"] = (sample["longitude"] - min_long) / (max_long - min_long) * 100
sample["latitude"] = (sample["latitude"] - min_lat) / (max_lat - min_lat) * 100


delta = 0.1
bounds = (0, 100)

db = TrajectoryDatabase.from_dataframe(sample)
data = np.array([traj.as_array() for traj in db.trajectories.values()])


def split_level_experiment(
    num_experiments_per_level=5, split_levels=[i**2 for i in range(2, 10)], eps=1.0
):

    dpm = DPM(data, bounds, eps, delta)
    results = {}
    for level in split_levels:
        dpm.num_split_levels = level
        for _ in range(num_experiments_per_level):
            passed = False
            retry = 0
            while not passed:
                try:
                    centers, _ = dpm.perform_clustering()
                    results.setdefault(level, []).append(len(centers))
                    passed = True
                except Exception as ex:
                    retry += 1
                    print(f"Error: {ex}, retrying {retry}")
    return results


def plot_split_level_experiment(results, eps=1.0, num_experiments_per_level=5):

    categories = [f"{l}" for l in results.keys()]
    values = [np.mean([size for size in sizes]) for sizes in results.values()]
    variances = [np.var([size for size in sizes]) for sizes in results.values()]

    plt.bar(categories, values, label="Average Number of Clusters")
    plt.plot(
        categories, variances, color="red", marker="o", linestyle="-", label="Variance"
    )
    plt.title(f"Epsilon: {eps}, Experiments per Level: {num_experiments_per_level}")
    plt.xlabel("Split Level")
    plt.ylabel("Average Number of Clusters")
    plt.legend()
    plt.show()


def eps_experiment(
    experiments_per_eps=5, eps=[0.1, 0.5, 1.0, 2.0, 5.0], split_levels=7
):

    results = {}
    for e in eps:
        dpm = DPM(data, bounds, e, delta)
        dpm.num_split_levels = split_levels
        for _ in range(experiments_per_eps):
            passed = False
            attempts = 0
            while not passed:
                try:
                    centers, _ = dpm.perform_clustering()
                    results.setdefault(e, []).append((len(centers), attempts))
                    passed = True
                except Exception as ex:
                    attempts += 1
                    print(f"Error: {ex}, {attempts} attemps")
    return results


def plot_eps_experiment(results, split_levels=7, experiments_per_eps=5):
    labels = [f"{l}" for l in results.keys()]
    avg_num_clusters = [
        np.mean([size for size, _ in result]) for result in results.values()
    ]
    error_rate = [
        np.mean([attempts for _, attempts in result]) for result in results.values()
    ]
    plt.title(
        f"Split Levels: {split_levels}, Experiments per Epsilon: {experiments_per_eps}"
    )
    plt.bar(labels, avg_num_clusters, label="Average Number of Clusters")
    plt.plot(
        labels,
        error_rate,
        color="red",
        marker="o",
        linestyle="-",
        label="Error Rate",
    )
    plt.xlabel("Epsilon")
    plt.ylabel("Average Number of Clusters")
    plt.legend()
    plt.show()


eps_results = eps_experiment(
    experiments_per_eps=10, eps=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0], split_levels=14
)
plot_eps_experiment(eps_results, experiments_per_eps=10, split_levels=14)
