import logging
import math
import time
import pytest
from pandas import DataFrame, read_csv
from sklearn.cluster import KMeans


from trajectory_clustering.data.read_db import csv_db_to_numpy
from trajectory_clustering.hua.hua import Hua

LOGGER = logging.getLogger(__name__)


@pytest.fixture(params=["4x5", "20x5", "500x5"])
def df(request) -> DataFrame:
    return read_csv(
        f"tests/data/fake-trajectories_{request.param}.csv",
    )


@pytest.fixture
def D(df):
    return csv_db_to_numpy(df)


@pytest.fixture
def p_opt(D) -> KMeans:
    n_clusters = max(2, round(D.shape[0] / 4))
    return KMeans(n_clusters=n_clusters).fit(D)


def test_phi_sub_optimal_inidividual(D, p_opt):
    labels, centers = p_opt.labels_, p_opt.cluster_centers_
    expected_size = min(D.shape[0] * (p_opt.n_clusters - 1), 1000)
    hua = Hua(1, expected_size, 1)
    start = time.time()
    result = hua._phi_sub_optimal_individual(D, (labels, centers))
    end = time.time()
    LOGGER.info(f"phi_sub_optimal_individual_np took: {round(end - start, 3)} seconds")

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


def test_phi_sub_optimal(D, p_opt):
    labels, centers = p_opt.labels_, p_opt.cluster_centers_
    expected_size = (
        1000
        if D.shape[0] > 10
        else int(
            sum(
                [
                    math.comb(D.shape[0], i) * math.pow(len(centers) - 1, i)
                    for i in range(1, D.shape[0] + 1)
                ]
            )
        )
    )
    hua = Hua(1, expected_size, 1)
    start = time.time()
    result = hua._phi_sub_optimal(D, (labels, centers))
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


@pytest.mark.parametrize("D", ["500x5"], indirect=True)
def test_s_kmeans_partitions(D):
    m = int(D.shape[0] / max(2, D.shape[0] / 4))
    hua = Hua(m, 1, 1)
    partitions = hua._s_kmeans_partitions(D)

    assert len(partitions) == D.shape[0]
    assert all(len(centers) == m for ((_, centers), _) in partitions)


@pytest.fixture
def generalized_locations(D):
    hua = Hua(m=int(max(2, D.shape[0] / 5)), phi=20, eps=1)
    return hua._dp_location_generalization(D)


def test_dp_location_generalization(D):
    m = int(max(2, D.shape[0] / 5))
    hua = Hua(m, phi=20, eps=1)
    start = time.time()
    generalized_locations = hua._dp_location_generalization(D)
    end = time.time()
    LOGGER.info(f"dp_location_generalization took: {end - start} seconds")
    assert len(generalized_locations) == m


def test_dp_release(D, generalized_locations):
    hua = Hua(m=int(max(2, D.shape[0] / 5)), phi=20, eps=1)
    start = time.time()
    trajects, counts = hua._dp_release(D, generalized_locations)
    end = time.time()
    LOGGER.info(f"dp_release took: {end - start} seconds")


def test_dp_hua(D):
    m = int(max(2, D.shape[0] / 5))
    hua = Hua(m, phi=20, eps=1)
    start = time.time()
    _, counts = hua.publish(D)
    end = time.time()
    LOGGER.info(f"dp_hua took: {end - start} seconds")
    assert sum(counts) >= m
