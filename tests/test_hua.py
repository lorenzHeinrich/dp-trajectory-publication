import logging
import math
import random
import time
import pytest
from numpy import array
from pandas import DataFrame, read_csv
from sklearn.cluster import KMeans
from trajectory_clustering.hua import (
    ClusteringResult,
    location_generalization,
    phi_sub_optimal,
    phi_sub_optimal_inidividual,
    s_kmeans_partitions,
)
from trajectory_clustering.trajectory import (
    SpatioTemporalPoint,
    TrajectoryDatabase,
    euclidean_distance,
)

LOGGER = logging.getLogger(__name__)


def test_euclidean_distance():
    x = array([0, 0])
    y = array([3, 4])
    assert euclidean_distance(x, y) == 5
    assert euclidean_distance(x, x) == 0
    assert euclidean_distance(x, y) == euclidean_distance(y, x)


@pytest.fixture(params=["4x5", "20x5", "500x5"])
def db(request) -> DataFrame:
    return read_csv(
        f"tests/data/fake-trajectories_{request.param}.csv",
    )


@pytest.fixture
def clusters(db, timestamp) -> ClusteringResult:
    n_clusters = round(len(db) / 10)
    kmeans = KMeans(n_clusters=n_clusters)
    trajectory_db = to_trajectory_db(db[db["timestamp"] == timestamp])
    points = [[p.x, p.y] for p in trajectory_db.trajectories]
    kmeans.fit(points)
    return ClusteringResult(
        trajectory_db.trajectories, kmeans.labels_, kmeans.cluster_centers_
    )


@pytest.fixture
def timestamp(db) -> int:
    timestamps = set(db["timestamp"])
    rand_index = random.randint(0, len(timestamps) - 1)
    return list(timestamps)[rand_index]


def to_trajectory_db(df: DataFrame) -> TrajectoryDatabase:
    return TrajectoryDatabase(
        [
            SpatioTemporalPoint(row[0], row[1], float(row[2]), float(row[3]))
            for row in df.itertuples(index=False)
        ]
    )


def test_phi_sub_optimal_inidividual(db, clusters, timestamp):
    points = db[db["timestamp"] == timestamp]

    expected_size = min(len(points) * (clusters.n_clusters - 1), 1000)
    start = time.time()
    result = phi_sub_optimal_inidividual(
        clusters,
        expected_size,
    )
    end = time.time()
    LOGGER.info(f"phi_sub_optimal_inidividual took: {end - start} seconds")
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


def test_phi_sub_optimal(db, clusters, timestamp):
    points = db[db["timestamp"] == timestamp]
    n_points = len(points)
    expected_size = (
        500
        if n_points > 10
        else int(
            sum(
                [
                    math.comb(n_points, i) * math.pow(clusters.n_clusters - 1, i)
                    for i in range(1, n_points + 1)
                ]
            )
        )
    )
    start = time.time()
    result = phi_sub_optimal(
        clusters,
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
    assert all(not duplicate(x, y) for x, y in pairs)

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
    db_t0 = db[db["timestamp"] == 0]
    trajectory_db = to_trajectory_db(db_t0)
    partitions = s_kmeans_partitions(trajectory_db.trajectories, 20)

    assert len(partitions) == len(trajectory_db.trajectories)
    assert all(len(p.cluster_centers) == 20 for p in partitions)


@pytest.mark.parametrize("db", ["20x5", "500x5"], indirect=True)
def test_location_generalization(db):
    trajectory_db = to_trajectory_db(db)
    location_generalization(trajectory_db, 2, 20, 0.1)
