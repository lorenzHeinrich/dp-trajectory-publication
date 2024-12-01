import logging
import math
import time
import pytest
from numpy import array
from pandas import DataFrame, read_csv
from sklearn.cluster import KMeans
from trajectory_clustering.hua import (
    ClusteringResult,
    phi_sub_optimal,
    phi_sub_optimal_inidividual,
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


timestamps = range(5)


@pytest.fixture(params=["4x5", "20x5", "1000x5"])
def db(request) -> DataFrame:
    return read_csv(
        f"tests/data/fake-trajectories_{request.param}.csv",
    )


@pytest.fixture
def clusters(db) -> dict[int, ClusteringResult]:
    result = {}
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters)
    for t in set(db["timestamp"]):
        points = db[db["timestamp"] == t][["longitude", "latitude"]]
        kmeans.fit(points)
        result[int(t)] = ClusteringResult(kmeans.labels_, kmeans.cluster_centers_)
    return result


def toTrajectoryDB(df: DataFrame) -> TrajectoryDatabase:
    return TrajectoryDatabase(
        [
            SpatioTemporalPoint(row[0], row[1], float(row[2]), float(row[3]))
            for row in df.itertuples(index=False)
        ]
    )


@pytest.mark.parametrize("t", timestamps)
def test_phi_sub_optimal_inidividual(db, clusters, t):
    points = db[db["timestamp"] == t]
    clusters_t = clusters[t]
    n_clusters = len(set(clusters_t.labels))
    expected_size = len(points) * (n_clusters - 1)
    result = phi_sub_optimal_inidividual(
        toTrajectoryDB(points),
        clusters_t,
        expected_size,
    )
    pairs = [
        (result[i], result[j])
        for i in range(len(result))
        for j in range(i + 1, len(result))
    ]
    # no duplicates in terms of id and cluster
    assert all(m1.id != m2.id or m1.cluster != m2.cluster for m1, m2 in pairs)
    # should have n * (k - 1) elements, where n is the number of points and k the number of clusters
    assert len(result) == expected_size

    # sorted ascendingly by distance
    assert all(
        result[i].distance <= result[i + 1].distance for i in range(len(result) - 1)
    )


@pytest.mark.parametrize("t", timestamps)
def test_phi_sub_optimal(db, clusters, t):
    points = db[db["timestamp"] == t]
    cluster_t = clusters[t]
    n_clusters = len(set(cluster_t.labels))
    n_points = len(points)
    expected_size = int(
        min(
            sum(
                [
                    math.comb(n_points, i) * math.pow(n_clusters - 1, i)
                    for i in range(1, n_points + 1)
                ]
            ),
            500,
        )
    )
    start = time.time()
    result = phi_sub_optimal(
        toTrajectoryDB(points),
        clusters[t],
        expected_size,
    )
    end = time.time()
    LOGGER.info(f"phi_sub_optimal took: {end - start} seconds")
    # no repeated modifications of individual points in a modification
    assert all(
        len([m.id for m in mod]) == len(set([m.id for m in mod])) for mod in result
    )

    # no duplicate modifications
    sort_by_id_cluster = lambda x: (x.id, x.cluster)
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
