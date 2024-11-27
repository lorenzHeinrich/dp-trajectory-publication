from datetime import datetime
from numpy import array
import pytest
from trajectory_clustering.hua import (
    ClusteringResult,
    SpatioTemporalPoint,
    TrajectoryDatabase,
    euclidean_distance,
    phi_sub_optimal_inidividual,
)


def test_euclidean_distance():
    x = array([0, 0])
    y = array([3, 4])
    assert euclidean_distance(x, y) == 5
    assert euclidean_distance(x, x) == 0
    assert euclidean_distance(x, y) == euclidean_distance(y, x)


@pytest.fixture
def simple_cluster():
    return {
        "points": [[1, 1], [1, 2], [3, 1], [3, 2]],
        "labels": [0, 0, 1, 1],
        "centers": [[1, 1.5], [3, 3.5]],
    }


def test_phi_sub_optimal_inidividual_simple(simple_cluster):
    database = TrajectoryDatabase(
        [
            SpatioTemporalPoint(i, datetime.now(), *point)
            for i, point in enumerate(simple_cluster["points"])
        ]
    )
    p_opt = ClusteringResult(
        labels=simple_cluster["labels"],
        cluster_centers=simple_cluster["centers"],
    )
    result = phi_sub_optimal_inidividual(database, p_opt, 4)
