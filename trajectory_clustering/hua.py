from numpy import array, float64, int64
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from trajectory_clustering.trajectory import SpatioTemporalPoint, TrajectoryDatabase


class ClusteringResult:
    def __init__(
        self,
        trajectories: list[SpatioTemporalPoint],
        labels: list[int] | NDArray[int64],
        cluster_centers: list[list[float]] | NDArray[float64],
    ) -> None:
        self.n_clusters = len(cluster_centers)
        self.cluster_labels = set(labels)
        self.cluster_centers = array(cluster_centers, dtype=float64)
        self.clusters: dict[int, dict[int, SpatioTemporalPoint]] = dict()
        for l, p in zip(labels, trajectories):
            self.clusters.setdefault(l, dict())[p.id] = p

    @property
    def labeled_trajectories(self):
        return [(l, p) for l, ps in self.clusters.items() for p in ps.values()]

    def __repr__(self) -> str:
        return f"ClusteringResult({self.clusters!r}, {self.cluster_centers!r})"

    def apply_modification(self, modification):
        p = self.clusters[modification.from_cluster].pop(modification.id)
        self.clusters[modification.to_cluster][p.id] = p


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


def s_k_means_partitions(
    database: TrajectoryDatabase, m: int
) -> list[ClusteringResult]:
    kmeans = KMeans(n_clusters=m)
    partitions = []
    for i in range(len(database.trajectories)):
        trajectories_ = database.trajectories[0:i] + database.trajectories[i + 1 :]
        kmeans.fit([[p.x, p.y] for p in trajectories_])
        partitions.append(
            ClusteringResult(trajectories_, kmeans.labels_, kmeans.cluster_centers_)
        )

    return partitions


def location_generalization(database: TrajectoryDatabase, m, phi):
    by_time = {}
    for location in database.trajectories:
        by_time[location.timestamp] = location

    kmeans = KMeans(n_clusters=m)
    for time, location in by_time.values():
        partitions = s_k_means_partitions(location, m)

        kmeans.fit([[p.x, p.y] for p in location.trajectories])
        p_opt = ClusteringResult(
            location.trajectories, kmeans.labels_, kmeans.cluster_centers_
        )

        modifications = phi_sub_optimal(p_opt, phi)
        modified_p_opt = [p_opt.apply_modification(m) for m in modifications]
