import pandas as pd
import pytest

from trajectory_clustering.cluster import kmeans_partitioning
from trajectory_clustering.trajectory import TrajectoryDatabase


@pytest.fixture
def db():
    return TrajectoryDatabase.from_dataframe(
        pd.read_csv("tests/data/fake-trajectories_4x5.csv")
    )


def test_kmeans_partioning(db):
    result = kmeans_partitioning(db, 2)

    expected_clusters = set([(0, 3), (1, 2)])
    clusters = set(
        [
            tuple([id for id in trajs.keys()])
            for trajs in result.labled_trajectories.values()
        ]
    )
    assert clusters == expected_clusters
