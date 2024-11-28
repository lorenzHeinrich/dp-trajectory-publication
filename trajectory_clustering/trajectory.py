from datetime import datetime
from numpy import array, floating, shape
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

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SpatioTemporalPoint):
            return False
        return (
            self.id == value.id
            and self.timestamp == value.timestamp
            and self.x == value.x
            and self.y == value.y
        )

    def distance(self, point: NDArray[floating]) -> float:
        assert shape(point) == (2,)
        return euclidean_distance(array([self.x, self.y]), point)


class TrajectoryDatabase:
    def __init__(
        self,
        trajectories: list[SpatioTemporalPoint],
    ) -> None:
        self.trajectories = trajectories

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TrajectoryDatabase):
            return False
        return self.trajectories == value.trajectories
