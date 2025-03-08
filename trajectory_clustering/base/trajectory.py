from datetime import datetime
import time
from turtle import st
from typing import overload
from numpy import array, floating, shape
from numpy.typing import NDArray
from scipy import linalg


def euclidean_distance(x: NDArray[floating], y: NDArray[floating]) -> floating:
    assert shape(x) == shape(y), "arrays should have same shape"
    assert len(shape(x)) == 1, "arrays should be one dimensional"
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

    def distance(self, other: "Location") -> floating:
        return euclidean_distance(array([self.x, self.y]), array([other.x, other.y]))


class STPoint:
    def __init__(self, timestamp: datetime | int, location: Location) -> None:
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

    def distance(self, other: "STPoint"):
        return self.location.distance(other.location)


class Trajectory:
    def __init__(self, id: int, st_points: list[STPoint]) -> None:
        self.id = id
        self.st_points = dict(
            (p.timestamp, p) for p in sorted(st_points, key=lambda x: x.timestamp)
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Trajectory):
            return False
        return self.id == value.id and self.st_points == value.st_points

    def __repr__(self) -> str:
        return f"Trajectory({self.id!r}, {self.st_points!r})"

    def as_array(self) -> NDArray[floating]:
        return array(
            [[p.location.x, p.location.y] for p in self.st_points.values()]
        ).reshape(-1)

    @classmethod
    def from_array(
        cls,
        flat_locations: NDArray[floating],
        timestamps: set[datetime | int],
        id: int | None = None,
    ):
        st_points = [
            STPoint(t, Location(location[0], location[1]))
            for t, location in zip(timestamps, flat_locations.reshape(-1, 2))
        ]
        return Trajectory(
            id=hash(tuple(st_points)) if id is None else id,
            st_points=st_points,
        )

    def distance(self, other: "Trajectory"):
        assert len(self.st_points) == len(
            other.st_points
        ), "trajectories should have same length"
        return euclidean_distance(self.as_array(), other.as_array())


class TrajectoryDatabase:
    def __init__(
        self, timestamps: set[datetime | int], trajectories: list[Trajectory] = []
    ) -> None:
        self.trajectories = dict((t.id, t) for t in trajectories)
        self.length = len(timestamps)
        self.timestamps = timestamps
        self.size = len(trajectories)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TrajectoryDatabase):
            return False
        return self.trajectories == value.trajectories

    def __repr__(self) -> str:
        return (
            f"TrajectoryDatabase({self.size!r}, {self.length!r}, {self.trajectories!r})"
        )

    def copy(self):
        return TrajectoryDatabase(self.timestamps, list(self.trajectories.values()))

    def remove(self, id: int):
        del self.trajectories[id]
        self.size -= 1
        return self

    def append(self, trajs: list[Trajectory]) -> None:
        for traj in trajs:
            self.trajectories[traj.id] = traj
        self.size += len(trajs)

    def keys(self):
        return self.trajectories.keys()

    @classmethod
    def from_dataframe(
        cls, df, id="id", timestamp="timestamp", x="longitude", y="latitude"
    ):
        trajectories = []
        timestamps = set(df[timestamp])
        for key in df[id].unique():
            # start = time.time()
            traj_data = df[df[id] == key]
            st_points = [
                STPoint(
                    timestamp=row[timestamp],
                    location=Location(row[x], row[y]),
                )
                for _, row in traj_data.iterrows()
            ]
            assert len(st_points) == len(
                timestamps
            ), "Trajectory should have points for all timestamps"
            trajectories.append(Trajectory(key, st_points))
            # end = time.time()
            # print(f"Trajectory {key} took {end - start} seconds")

        return TrajectoryDatabase(set(df[timestamp]), trajectories)
