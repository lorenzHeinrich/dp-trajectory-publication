from numpy import array, float64, int64
from numpy.typing import NDArray

from trajectory_clustering.trajectory import TrajectoryDatabase


class ClusteringResult:
    def __init__(
        self,
        labels: list[int] | NDArray[int64],
        cluster_centers: list[list[float]] | NDArray[float64],
    ) -> None:
        self.labels = array(labels, dtype=int64)
        self.cluster_centers = array(cluster_centers, dtype=float64)

    def __repr__(self) -> str:
        return f"ClusteringResult({self.labels!r}, {self.cluster_centers!r})"


class Modification:
    def __init__(
        self,
        id: int,
        cluster: int,
        distance: float,
    ) -> None:
        self.id = id
        self.cluster = cluster
        self.distance = distance

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
            modifications.append(Modification(p.id, int(c), distance))

    modifications.sort(key=lambda x: x.distance)
    return modifications[0:phi]


def phi_sub_optimal(
    database: TrajectoryDatabase,
    p_opt: ClusteringResult,
    phi: int,
):
    indiv_mods = phi_sub_optimal_inidividual(database, p_opt, phi)
    result = [[indiv_mods[0]]]

    for indiv_mod in indiv_mods[1:]:
        tmp = [[indiv_mod]]

        for mod in result:
            if all(m.id != indiv_mod.id for m in mod):
                new_mod = mod.copy()
                new_mod.append(indiv_mod)
                tmp.append(new_mod)

        result = result + tmp

    return result
