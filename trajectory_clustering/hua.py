from datetime import datetime
import math
import secrets
from typing import Callable
from numpy import float64, floating, int64

from trajectory_clustering.cluster import Partition, kmeans_partitioning
from trajectory_clustering.dp_mechanisms import (
    exponential_mechanism,
    laplace_mechanism,
    random_int,
)
from trajectory_clustering.trajectory import (
    Location,
    STPoint,
    Trajectory,
    TrajectoryDatabase,
)


class Modification:
    def __init__(
        self,
        id: int,
        from_cluster: int | int64,
        to_cluster: int | int64,
        distance: float | floating,
    ) -> None:
        self.id = id
        self.from_cluster = int64(from_cluster)
        self.to_cluster = int64(to_cluster)
        self.distance = float64(distance)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Modification):
            return False
        return (
            self.id == value.id
            and self.from_cluster == value.from_cluster
            and self.to_cluster == value.to_cluster
            and self.distance == value.distance
        )

    def __repr__(self) -> str:
        return f"Modification({self.id!r}, {self.from_cluster!r}, {self.to_cluster!r} {self.distance!r})"


def phi_sub_optimal_inidividual(
    p_opt: Partition,
    phi: int,
) -> list[Modification]:
    modifications: list[Modification] = []

    for l, group in p_opt.labled_trajectories.items():
        for id, traj in group.items():
            dis_opt = traj.distance(p_opt.mean_trajectories[l])
            for l_ in p_opt.labled_trajectories.keys() - {l}:
                dis_mod = traj.distance(p_opt.mean_trajectories[l_])
                modifications.append(Modification(id, l, l_, dis_mod - dis_opt))

    return sorted(modifications, key=lambda x: float(x.distance))[0:phi]


def phi_sub_optimal(
    p_opt: Partition,
    phi: int,
) -> list[list[Modification]]:
    indiv_mods = phi_sub_optimal_inidividual(p_opt, phi)
    result = [[indiv_mods[0]]]

    for indiv_mod in indiv_mods[1:]:
        tmp = [[indiv_mod]]

        for mod in result:
            if not any(indiv_mod.id == m.id for m in mod):
                new_mod = mod.copy() + [indiv_mod]
                tmp.append(new_mod)

        result += tmp
        result.sort(key=lambda x: sum(float(m.distance) for m in x))
        if len(result) > phi:
            result = result[0:phi]
    return result


def s_kmeans_partitions(db: TrajectoryDatabase, m: int) -> list[Partition]:
    partitions = []
    for id in db.keys():
        db_ = db.copy().remove(id)
        partitions.append(kmeans_partitioning(db_, m))
    return partitions


def mean_distance(partition: Partition) -> float:
    sum = 0
    for l, group in partition.labled_trajectories.items():
        group_sum = 0
        for traj in group.values():
            group_sum += traj.distance(partition.mean_trajectories[l])
        sum += group_sum / len(group) if len(group) > 0 else 0
    return float(sum / partition.n_labels)


def dp_location_generalization(db: TrajectoryDatabase, m, phi, eps) -> Partition:
    p_opt = kmeans_partitioning(db, m)
    modifications = phi_sub_optimal(p_opt, phi)
    s_partitions = s_kmeans_partitions(db, m)

    p_opt_modified = []
    for mods in modifications:
        p_opt_ = p_opt.copy()
        for mod in mods:
            p_opt_.move(
                mod.id, mod.from_cluster, mod.to_cluster
            )  # move with recalcualtion of mean, not clear in the paper
        p_opt_modified.append(p_opt_)

    partitions = p_opt_modified + [p_opt] + s_partitions

    p_opt_mean_dist = mean_distance(p_opt)
    utility_score = lambda x: p_opt_mean_dist / mean_distance(
        x
    )  # x.mean_distance() may be zero, but highly unlikely for real-world data

    partition = exponential_mechanism(partitions, utility_score, 1, eps)

    return partition


def laplace_integral(a, b, eps):
    return laplace_indefinite_integral(b, eps) - laplace_indefinite_integral(a, eps)


def laplace_indefinite_integral(x, eps):
    return -1 / 2 * math.exp(-x / eps) if x >= 0 else 1 / 2 * math.exp(x / eps)


def draw_trajectory(
    location_universe_by_time: dict[datetime | int, set[Location]],
    timestamps: set[datetime | int],
) -> Trajectory:
    st_points = []
    for t in timestamps:
        st_points.append(STPoint(t, secrets.choice(list(location_universe_by_time[t]))))
    return Trajectory(hash(tuple(st_points)), st_points)


def make_noisy_counts(partition: Partition, eps: float):
    query: Callable[[Partition], list[int]] = lambda x: [
        len(x.labled_trajectories[l]) for l in x.labels
    ]
    return zip(
        sorted(
            laplace_mechanism(partition, query, sensitivity=1, eps=eps),
            reverse=True,
        ),
        partition.labels,
    )


def num_is_by_intervals(
    location_universe: dict[datetime | int, set[Location]],
    noisy_counts: list[int],
    partition,
    eps: float,
) -> dict[tuple[int, int], int]:
    num_possible_trajectories = math.prod([len(l) for l in location_universe.values()])
    num_remaining_possible_traj = num_possible_trajectories - len(
        partition.mean_trajectories
    )
    return {
        (a, b): round(num_remaining_possible_traj * laplace_integral(a, b, eps))
        for a, b in zip(noisy_counts[1:], noisy_counts[:-1])
        if abs(a - b) > 1
    }


def dp_release(
    db: TrajectoryDatabase, partition: Partition, eps: float
) -> list[tuple[int, Trajectory]]:
    # filter out non-positive counted trajectories
    noisy_counts = list(filter(lambda x: x[0] > 0, make_noisy_counts(partition, eps)))
    location_universe = partition.location_universes_by_time()
    num_is = num_is_by_intervals(
        location_universe, [c for c, _ in noisy_counts], partition, eps
    )

    release: list[tuple[int, Trajectory]] = []
    acc_count = 0
    for i, (c, l) in enumerate(noisy_counts):
        noisy_count = min(db.size - acc_count, c)
        release.append((noisy_count, partition.mean_trajectories[l]))
        acc_count += noisy_count
        if acc_count >= db.size:
            break

        if i + 1 >= len(noisy_counts):
            continue
        interval = (noisy_counts[i + 1][0], c)
        if abs(interval[0] - interval[1]) <= 1:
            continue

        num_i = num_is[interval]
        while num_i > 0 and acc_count < db.size:
            traj = draw_trajectory(location_universe, db.timestamps)
            rand_noisy_count = min(
                db.size - acc_count, random_int(interval[0], interval[1])
            )
            release.append((rand_noisy_count, traj))
            num_i -= 1
            acc_count += rand_noisy_count

        if acc_count >= db.size:
            break

    # if we still have not enough trajectories, fill up with noisy trajectories
    if acc_count < db.size:
        smalest_noisy_count = noisy_counts[-1][0]
        while acc_count < db.size:
            traj = draw_trajectory(location_universe, db.timestamps)
            rand_noisy_count = min(
                db.size - acc_count, random_int(0, smalest_noisy_count)
            )
            release.append((rand_noisy_count, traj))
            acc_count += rand_noisy_count

    return release
