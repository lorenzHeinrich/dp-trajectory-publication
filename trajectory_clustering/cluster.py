from datetime import datetime
from numpy import array, int64, mean
from sklearn.cluster import KMeans

from trajectory_clustering.trajectory import (
    Location,
    STPoint,
    Trajectory,
    TrajectoryDatabase,
)


class Partition:
    def __init__(
        self,
        labled_trajectories: dict[int64, dict[int, Trajectory]],
        mean_trajectories: dict[int64, Trajectory],
    ) -> None:
        self.labled_trajectories = labled_trajectories
        self.mean_trajectories = mean_trajectories

    def __repr__(self) -> str:
        return f"Partition({self.labled_trajectories!r},{self.mean_trajectories!r})"

    @property
    def n_labels(self) -> int:
        return len(self.labled_trajectories)

    @property
    def labels(self) -> list[int64]:
        return list(self.labled_trajectories.keys())

    def location_universes_by_time(self) -> dict[datetime | int, list[Location]]:
        universe: dict[datetime | int, list[Location]] = {}
        for traj in self.mean_trajectories.values():
            for t, p in traj.st_points.items():
                universe.setdefault(t, []).append(p.location)
        return universe

    def location_universe(self) -> list[Location]:
        locations = []
        for _, l in self.location_universes_by_time().items():
            locations += l
        return locations

    def copy(self):
        return Partition(
            {label: dict(trajs) for label, trajs in self.labled_trajectories.items()},
            {label: traj for label, traj in self.mean_trajectories.items()},
        )

    def move(self, id: int, from_label: int64, to_label: int64):
        traj = self.labled_trajectories[from_label].pop(id)
        if to_label not in self.labled_trajectories:
            self.labled_trajectories[to_label] = {}
            self.mean_trajectories[to_label] = traj
        self.labled_trajectories[to_label][id] = traj
        self.recalc_mean(to_label)
        if len(self.labled_trajectories[from_label]) == 0:
            del self.labled_trajectories[from_label]
            del self.mean_trajectories[from_label]
        else:
            self.recalc_mean(from_label)

    def recalc_mean(self, label: int64):
        locations_arr = array(
            [traj.as_array() for traj in self.labled_trajectories[label].values()]
        )
        mean_locations_arr = mean(locations_arr, axis=0)

        self.mean_trajectories[label] = Trajectory.from_array(
            id=self.mean_trajectories[label].id,
            flat_locations=mean_locations_arr,
            timestamps=sorted(self.mean_trajectories[label].st_points.keys()),
        )


def kmeans_partitioning(db: TrajectoryDatabase, m: int):
    transformed = array([traj.as_array() for traj in db.trajectories.values()])

    kmeans = KMeans(n_clusters=m).fit(transformed)
    labeled_trajectories: dict[int64, dict[int, Trajectory]] = {}
    for label, traj in zip(kmeans.labels_, db.trajectories.values()):
        labeled_trajectories.setdefault(label, {}).update({traj.id: traj})

    mean_trajectories = {}
    for label, center in zip(set(kmeans.labels_), kmeans.cluster_centers_):
        locations = [Location(x, y) for x, y in center.reshape(-1, 2)]
        st_points = [STPoint(t, l) for t, l in zip(db.timestamps, locations)]
        mean_id = hash(tuple(traj.id for traj in labeled_trajectories[label].values()))
        mean_trajectories[label] = Trajectory(mean_id, st_points)

    return Partition(labeled_trajectories, mean_trajectories)
