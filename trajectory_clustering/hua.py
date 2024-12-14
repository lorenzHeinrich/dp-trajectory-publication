from numpy import float64, floating, int64, mean

from trajectory_clustering.cluster import Partition, kmeans_partitioning
from trajectory_clustering.dp_mechanisms import exponential_mechanism
from trajectory_clustering.trajectory import TrajectoryDatabase


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
):
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
):
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


def mean_distance(partition: Partition):
    sum = 0
    for l, group in partition.labled_trajectories.items():
        group_sum = 0
        for traj in group.values():
            group_sum += traj.distance(partition.mean_trajectories[l])
        sum += group_sum / len(group) if len(group) > 0 else 0
    return sum / partition.n_labels


def location_generalization(db: TrajectoryDatabase, m, phi, eps):
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
