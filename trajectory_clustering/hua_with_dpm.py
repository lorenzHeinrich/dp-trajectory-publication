from datetime import datetime
import math
from diffprivlib.mechanisms import Laplace
import numpy as np

from dpm.dpm import DPM
import pandas as pd
from trajectory_clustering.dp_mechanisms import random_int
from trajectory_clustering.hua import (
    draw_trajectory,
    laplace_integral,
)
from trajectory_clustering.trajectory import Location, Trajectory, TrajectoryDatabase


def dpm_location_generalization(
    eps: float, delta: float, db: TrajectoryDatabase, bounds: tuple[float, float]
) -> list[tuple[Trajectory, list[int]]]:
    data = np.array([traj.as_array() for traj in db.trajectories.values()])
    dpm = DPM(data, bounds, eps, delta)

    centers, clusters = dpm.perform_clustering()

    generalized = [
        (
            Trajectory.from_array(center, db.timestamps),
            cluster,
        )
        for center, cluster in zip(centers, clusters)
    ]

    return generalized


def noisy_release(db: TrajectoryDatabase, generalized, eps):
    laplace = Laplace(epsilon=eps, sensitivity=1)  # revisit sensitivity
    true_counts = [len(cluster) for _, cluster in generalized]
    print(f"true_counts {true_counts}")
    noisy_counts = [int(laplace.randomise(c)) for c in true_counts]
    print(f"noisy counts: {noisy_counts}")
    noisy_mean_trajectories = sorted(
        [
            (traj, noisy_count)
            for (traj, _), noisy_count in zip(generalized, noisy_counts)
            if noisy_count > 0
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    release = TrajectoryDatabase(timestamps=db.timestamps)
    release_size = 0

    universe = location_universe_by_time(noisy_mean_trajectories)
    universe_size = math.prod([len(l) for l in universe.values()]) - sum(noisy_counts)

    for (traj, c_1), (_, c_2) in zip(
        noisy_mean_trajectories, (noisy_mean_trajectories + [(None, 0)])[1:]
    ):
        print(f"adding {c_1} of trajectory {traj.id}")
        release.append([traj] * c_1)
        release_size += c_1
        if release_size >= db.size:
            break

        num_i = universe_size * laplace_integral(c_2, c_1, eps)
        while num_i > 0 and release_size < db.size:
            rand_traj = draw_trajectory(universe, db.timestamps)
            rand_count = random_int(c_2, c_1)
            print(f"adding {rand_count} of random trajectory {rand_traj.id}")
            release.append([rand_traj] * rand_count)
            release_size += rand_count
            num_i -= 1

    return release


def location_universe_by_time(generalized) -> dict[datetime | int, set[Location]]:
    universe = {}
    for mean_traj, _ in generalized:
        for t, p in mean_traj.st_points.items():
            universe.setdefault(t, set()).add(p.location)
    return universe


def dpm_hua(db, eps_1, delta_1, eps_2, bounds):
    generalized = dpm_location_generalization(eps_1, delta_1, db, bounds)
    return noisy_release(db, generalized, eps_2)
