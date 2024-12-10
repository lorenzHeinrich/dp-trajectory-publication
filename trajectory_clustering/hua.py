from numpy import array, float64, int64
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from datetime import datetime

from trajectory_clustering.mechanisms import exponential_mechanism
from trajectory_clustering.trajectory import (
    STPoint,
    TrajectoryDatabase,
)


class Modification:
    def __init__(
        self,
        id: int,
        from_cluster: int,
        to_cluster: int,
        distance: float,
    ) -> None:
        self.id = id
        self.from_cluster = from_cluster
        self.to_cluster = to_cluster
        self.distance = distance

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


class ClusteringResult:
    def __init__(
        self,
        trajectories: list[STPoint],
        labels: list[int] | NDArray[int64],
        cluster_centers: list[list[float]] | NDArray[float64],
    ) -> None:
        self.n_clusters = len(cluster_centers)
        self.n_trajectories = len(trajectories)
        self.cluster_labels = set(labels)
        self.cluster_centers = array(cluster_centers, dtype=float64)
        self.clusters: dict[int, dict[int, STPoint]] = dict()
        for l, p in zip(labels, trajectories):
            self.clusters.setdefault(l, dict())[p.id] = p

    @property
    def labeled_trajectories(self):
        return [(l, p) for l, ps in self.clusters.items() for p in ps.values()]

    def __repr__(self) -> str:
        return f"ClusteringResult({self.clusters!r}, {self.cluster_centers!r})"

    def apply_modification(
        self, modification: list[Modification]
    ) -> "ClusteringResult":
        modified = self.__copy()
        for mod in modification:
            p = modified.clusters[mod.from_cluster].pop(mod.id)
            modified.clusters[mod.to_cluster][p.id] = p
        return modified

    def mean_distance(self) -> float:
        sum = 0.0
        for c, trajectories in self.clusters.items():
            for t in trajectories.values():
                sum += t.distance(self.cluster_centers[c])
        return 1 / self.n_trajectories * sum  # deviation from paper

    def __copy(self) -> "ClusteringResult":
        labels_trajectories = array([[l, p] for l, p in self.labeled_trajectories])
        return ClusteringResult(
            labels_trajectories[:, 1],  # type: ignore
            labels_trajectories[:, 0],
            self.cluster_centers,
        )


def phi_sub_optimal_inidividual(
    p_opt: ClusteringResult,
    phi: int,
):
    modifications: list[Modification] = []

    for k, points in p_opt.clusters.items():
        for p in points.values():
            k_center = p_opt.cluster_centers[k]

            for c in p_opt.cluster_labels - {k}:
                c_center = p_opt.cluster_centers[c]
                distance = p.distance(c_center) - p.distance(k_center)
                modifications.append(Modification(p.id, k, c, distance))

    modifications.sort(key=lambda x: x.distance)
    return modifications[0:phi]


def phi_sub_optimal(
    p_opt: ClusteringResult,
    phi: int,
):
    indiv_mods = phi_sub_optimal_inidividual(p_opt, phi)
    result = [[indiv_mods[0]]]

    for indiv_mod in indiv_mods[1:]:
        tmp = [[indiv_mod]]

        for mod in result:
            if all(indiv_mod.id != m.id for m in mod):
                new_mod = mod.copy()
                new_mod.append(indiv_mod)
                tmp.append(new_mod)

        result += tmp
        result.sort(key=lambda x: sum(m.distance for m in x))
        if len(result) > phi:
            result = result[0:phi]
    return result


def s_kmeans_partitions(trajectories: list[STPoint], m: int) -> list[ClusteringResult]:
    kmeans = KMeans(n_clusters=m)
    partitions = []
    for i in range(len(trajectories)):
        trajectories_ = trajectories[0:i] + trajectories[i + 1 :]
        kmeans.fit([[p.x, p.y] for p in trajectories_])
        partitions.append(
            ClusteringResult(trajectories_, kmeans.labels_, kmeans.cluster_centers_)
        )

    return partitions


def location_generalization(
    database: TrajectoryDatabase, m, phi, eps
) -> dict[datetime, list[STPoint]]:
    by_time: dict[datetime, list[STPoint]] = {}
    for locations in database.trajectories:
        by_time.setdefault(locations.timestamp, []).append(locations)

    lengths = [len(locations) for locations in by_time.values()]
    assert all(length == lengths[0] for length in lengths)

    n_trajectories_per_t = lengths[0]
    assert m <= n_trajectories_per_t - 2

    generalized: dict[datetime, list[STPoint]] = {}
    kmeans = KMeans(n_clusters=m)
    for time, locations in by_time.items():
        partitions = s_kmeans_partitions(locations, m)

        kmeans.fit([[p.x, p.y] for p in locations])
        p_opt = ClusteringResult(locations, kmeans.labels_, kmeans.cluster_centers_)
        partitions.append(p_opt)

        modifications = phi_sub_optimal(p_opt, phi)
        modified_p_opt = [p_opt.apply_modification(m) for m in modifications]
        partitions += modified_p_opt

        p_opt_mean_dist = p_opt.mean_distance()
        utility_score = (
            lambda x: p_opt_mean_dist / x.mean_distance()
        )  # x.mean_distance() may be zero, but highly unlikely for real-world data

        p = exponential_mechanism(partitions, utility_score, 1, eps)
        for cluster, locations in p.clusters.items():
            c = p.cluster_centers[cluster]
            id = hash(locations.values())
            generalized.setdefault(time, []).append(
                STPoint(id, time, float(c[0]), float(c[1]))
            )

    return generalized
