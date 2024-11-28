from numpy import array
from pandas import read_csv
import pytest
from trajectory_clustering.hua import (
    ClusteringResult,
    Modification,
    TrajectoryDatabase,
    phi_sub_optimal_inidividual,
)
from trajectory_clustering.trajectory import (
    SpatioTemporalPoint,
    euclidean_distance,
)


def test_euclidean_distance():
    x = array([0, 0])
    y = array([3, 4])
    assert euclidean_distance(x, y) == 5
    assert euclidean_distance(x, x) == 0
    assert euclidean_distance(x, y) == euclidean_distance(y, x)


@pytest.fixture
def db_t0():
    df = read_csv("tests/data/fake-trajectories_10x10.csv", parse_dates=["timestamp"])
    return TrajectoryDatabase(
        list(
            filter(
                lambda p: p.timestamp == "0",
                map(
                    lambda t: SpatioTemporalPoint(*t),
                    df.itertuples(index=False, name=None),
                ),
            )
        )
    )


@pytest.fixture
def clusters_t0():
    return ClusteringResult(
        labels=[2, 0, 0, 1],
        cluster_centers=[[0, 3.5], [8, 8], [7, 9]],
    )


@pytest.fixture
def all_modifications_t0():
    return [
        Modification(0, 1, 1.41),
        Modification(3, 2, 1.41),
        Modification(1, 2, 8.10),
        Modification(1, 1, 8.44),
        Modification(2, 2, 8.72),
        Modification(0, 0, 8.90),
        Modification(2, 1, 8.93),
        Modification(3, 0, 9.18),
    ]


def test_phi_sub_optimal_inidividual_simple(db_t0, clusters_t0, all_modifications_t0):
    phi = len(all_modifications_t0)
    result = phi_sub_optimal_inidividual(db_t0, clusters_t0, phi)
    result = list(
        map(
            lambda m: Modification(
                m.id,
                m.cluster,
                round(m.distance, 2),
            ),
            result,
        )
    )
    assert result == all_modifications_t0
