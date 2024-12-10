from datetime import datetime
from numpy import array, floating, long, shape
from numpy.typing import NDArray
from scipy import linalg


def euclidean_distance(x: NDArray[floating], y: NDArray[floating]) -> floating:
    assert shape(x) == shape(y)
    eucl_dis = linalg.norm(x - y)
    return eucl_dis


class Location:
    def __init__(self, x: floating, y: floating):
        self.x = x
        self.y = y

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Location):
            return False
        return self.x == value.x and self.y == value.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"Location({self.x!r}, {self.y!r})"

    def distance(self, point: NDArray[floating]) -> floating:
        assert shape(point) == (2,)
        return euclidean_distance(array([self.x, self.y]), point)


class STPoint:
    def __init__(self, timestamp: datetime, location: Location) -> None:
        self.timestamp = timestamp
        self.location = location

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, STPoint):
            return False
        return self.timestamp == value.timestamp and self.location == value.timestamp

    def __hash__(self) -> int:
        return hash((self.timestamp, self.location))

    def __repr__(self) -> str:
        return f"SpatioTemporalPoint({self.timestamp!r}, {self.location!r}"


class Trajectory:
    def __init__(self, id: long, st_points: list[STPoint]) -> None:
        self.id = id
        self.st_points = dict(
            (p.timestamp, p) for p in sorted(st_points, key=lambda x: x.timestamp)
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Trajectory):
            return False
        return self.id == value.id and self.st_points == value.st_points


class TrajectoryDatabase:
    def __init__(self, trajectories: list[Trajectory]) -> None:
        self.trajectories = trajectories

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TrajectoryDatabase):
            return False
        return self.trajectories == value.trajectories

    def __repr__(self) -> str:
        return f"TrajectoryDatabase({self.trajectories!r})"
