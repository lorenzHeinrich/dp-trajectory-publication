from datetime import datetime
from numpy import array, float64, floating, int64, shape
from numpy.typing import NDArray
from scipy import linalg


def euclidean_distance(x: NDArray[floating], y: NDArray[floating]) -> float:
    assert shape(x) == shape(y)
    eucl_dis = linalg.norm(x - y)
    assert isinstance(eucl_dis, float)
    return eucl_dis


class SpatioTemporalPoint:
    def __init__(
        self,
        id: int,
        timestamp: datetime,
        x: float,
        y: float,
    ) -> None:
        self.id = id
        self.timestamp = timestamp
        self.x = x
        self.y = y

    def distance(self, point: NDArray[floating]) -> float:
        assert shape(point) == (2,)
        return euclidean_distance(array([self.x, self.y]), point)


class TrajectoryDatabase:
    def __init__(
        self,
        trajectories: list[SpatioTemporalPoint],
    ) -> None:
        self.trajectories = trajectories


class ClusteringResult:
    def __init__(
        self,
        labels: list[int],
        cluster_centers: list[list[float]],
    ) -> None:
        self.labels = array(labels, dtype=int64)
        self.cluster_centers = array(cluster_centers, dtype=float64)


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

    modifications.sort(key=lambda x: x.distance)
    return modifications[0:phi]
