from numpy import array, float64, int64

from trajectory_clustering.trajectory import TrajectoryDatabase


class ClusteringResult:
    def __init__(
        self,
        labels: list[int | int64],
        cluster_centers: list[list[float | float64]],
    ) -> None:
        self.labels = array(labels, dtype=int64)
        self.cluster_centers = array(cluster_centers, dtype=float64)


class Modification:
    def __init__(
        self,
        id: int | int64,
        cluster: int | int64,
        distance: float | float64,
    ) -> None:
        self.id = int64(id)
        self.cluster = int64(cluster)
        self.distance = float64(distance)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Modification):
            return False
        return (
            self.id == value.id
            and self.cluster == value.cluster
            and self.distance == value.distance
        )

    def __repr__(self) -> str:
        return f"Modification({self.id!r}, {self.cluster!r}, {self.distance!r})"


def phi_sub_optimal_inidividual(
    database: TrajectoryDatabase,
    p_opt: ClusteringResult,
    phi: int,
):
    modifications: list[Modification] = []
    clusters = set(p_opt.labels)
    for k, p in zip(p_opt.labels, database.trajectories, strict=True):
        k_center = p_opt.cluster_centers[k]

        for c in clusters - {k}:
            c_center = p_opt.cluster_centers[c]
            distance = p.distance(c_center) - p.distance(k_center)
            modifications.append(Modification(p.id, c, distance))

    modifications.sort(key=lambda x: float(x.distance))
    return modifications[0:phi]
