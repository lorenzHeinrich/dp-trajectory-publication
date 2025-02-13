import logging
import math
import time
import pytest
from numpy import array
from pandas import DataFrame, read_csv

from trajectory_clustering.cluster import Partition, kmeans_partitioning
from trajectory_clustering.hua import (
    dp_hua,
    dp_location_generalization,
    dp_release,
    phi_sub_optimal,
    phi_sub_optimal_inidividual,
    s_kmeans_partitions,
)
from trajectory_clustering.trajectory import (
    Trajectory,
    TrajectoryDatabase,
    euclidean_distance,
)
from trajectory_clustering.visualize import stepwise_plot

LOGGER = logging.getLogger(__name__)


def test_euclidean_distance():
    x = array([0, 0])
    y = array([3, 4])
    assert euclidean_distance(x, y) == 5
    assert euclidean_distance(x, x) == 0
    assert euclidean_distance(x, y) == euclidean_distance(y, x)


@pytest.fixture(params=["4x5", "20x5", "500x5"])
def df(request) -> DataFrame:
    return read_csv(
        f"tests/data/fake-trajectories_{request.param}.csv",
    )


@pytest.fixture
def db(df) -> TrajectoryDatabase:
    return TrajectoryDatabase.from_dataframe(df)


@pytest.fixture
def partition(db) -> Partition:
    n_clusters = max(2, round(db.size / 4))
    return kmeans_partitioning(db, n_clusters)


def test_phi_sub_optimal_inidividual(db, partition):

    expected_size = min(db.size * (partition.n_labels - 1), 1000)
    start = time.time()
    result = phi_sub_optimal_inidividual(
        partition,
        expected_size,
    )
    end = time.time()
    LOGGER.info(f"phi_sub_optimal_inidividual took: {round(end - start, 3)} seconds")
    pairs = [
        (result[i], result[j])
        for i in range(len(result))
        for j in range(i + 1, len(result))
    ]
    # no duplicates in terms of id and cluster
    assert all(m1.id != m2.id or m1.to_cluster != m2.to_cluster for m1, m2 in pairs)
    # should have n * (k - 1) elements, where n is the number of points and k the number of clusters
    assert len(result) == expected_size

    # sorted ascendingly by distance
    assert all(
        result[i].distance <= result[i + 1].distance for i in range(len(result) - 1)
    )


def test_phi_sub_optimal(db, partition):
    expected_size = (
        1000
        if db.size > 10
        else int(
            sum(
                [
                    math.comb(db.size, i) * math.pow(partition.n_labels - 1, i)
                    for i in range(1, db.size + 1)
                ]
            )
        )
    )
    start = time.time()
    result = phi_sub_optimal(
        partition,
        expected_size,
    )
    end = time.time()
    LOGGER.info(f"phi_sub_optimal took: {end - start} seconds")
    # no repeated modifications of individual points in a modification
    assert all(
        len([m.id for m in mod]) == len(set([m.id for m in mod])) for mod in result
    )

    # no duplicate modifications
    sort_by_id_cluster = lambda x: (x.id, x.from_cluster, x.to_cluster)
    duplicate = lambda x, y: (
        sorted(x, key=sort_by_id_cluster) == sorted(y, key=sort_by_id_cluster)
    )
    pairs = [
        (result[i], result[j])
        for i in range(len(result))
        for j in range(i + 1, len(result))
    ]
    assert not any(duplicate(x, y) for x, y in pairs)

    # result should have sum_{i=1}^{n} (n choose i) * (k - 1)^i elements,
    # where n is the number of points and k the number of clusters.
    # We limit the expected size to 500, which is the paramter phi
    assert len(result) == expected_size

    # sorted ascendingly by sum of distances
    assert all(
        sum(m.distance for m in result[i]) <= sum(m.distance for m in result[i + 1])
        for i in range(len(result) - 1)
    )


@pytest.mark.parametrize("db", ["500x5"], indirect=True)
def test_s_kmeans_partitions(db):
    m = int(db.size / max(2, db.size / 4))
    partitions = s_kmeans_partitions(db, m)

    assert len(partitions) == db.size
    assert all(len(p.mean_trajectories) == m for p in partitions)


def test_location_generalization(db):
    partition = dp_location_generalization(db, int(max(2, db.size / 5)), 20, 0.1)

    stepwise_plot(partition)
    print(partition)


@pytest.fixture
def generalized_locations(db):
    return dp_location_generalization(db, int(max(2, db.size / 5)), phi=20, eps=1)


def test_dp_release(db: TrajectoryDatabase, generalized_locations: Partition):
    start = time.time()
    release: list[tuple[int, Trajectory]] = dp_release(db, generalized_locations, 1)
    end = time.time()
    LOGGER.info(f"dp_release took: {end - start} seconds")
    assert sum(n for n, _ in release) == db.size


def test_dp_hua(db):
    start = time.time()
    sanatized = dp_hua(db, 1.0, max(2, int(db.size / 5)), 20)
    end = time.time()
    LOGGER.info(f"dp_hua took: {end - start} seconds")
    assert sanatized.size == db.size
