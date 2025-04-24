import numpy as np

from scipy.spatial.distance import directed_hausdorff


def possibly_sometimes_inside(D, R, t_interval, uncertainty):
    """
    Calculate the number of trajectories that are possibly inside the circle at least once,
    considering a fixed uncertainty radius for each trajectory point.

    :param D: Original dataset
    :param R: Circle defined by center (x, y) and radius r
    :param t_interval: Time interval (tb, te)
    :param uncertainty: Fixed uncertainty radius for the entire trajectory, for each point
    :return: Number of trajectories that are possibly inside the circle at least once
    """
    tb, te = t_interval
    (x, y), r = R
    D_int = D[:, tb:te, :]

    distances = np.sqrt((D_int[:, :, 0] - x) ** 2 + (D_int[:, :, 1] - y) ** 2)

    # Check if the distance from each point to the center of the circle is
    # less than or equal to the radius plus the uncertainty for any point
    inside = np.any(distances <= (r + uncertainty), axis=1)

    return np.sum(inside)


def definitely_always_inside(D, R, t_interval, uncertainty):
    """
    Calculate the number of trajectories that are definitely always inside the circle,
    considering the full uncertainty radius around each trajectory point.

    :param D: Original dataset
    :param R: Circle defined by center (x, y) and radius r
    :param t_interval: Time interval (tb, te)
    :param uncertainty: Fixed uncertainty radius for each trajectory point
    :return: Number of trajectories that are definitely always inside the circle
    """
    tb, te = t_interval
    (x, y), r = R

    D_int = D.reshape(D.shape[0], -1, 2)[:, tb : te + 1, :]

    distances = np.sqrt((D_int[:, :, 0] - x) ** 2 + (D_int[:, :, 1] - y) ** 2)

    # Check if the distance from each point to the center of the circle is
    # less than or equal to the radius minus the uncertainty for all points
    inside = np.all(distances <= (r - uncertainty), axis=1)

    return np.sum(inside)


def range_query_distortion(D, D_pub, Q):
    """
    Compute the range query distortion between the original and published datasets.
    :param D: Original dataset
    :param D_pub: Published dataset
    :param Q: Query function
    :return: Range query distortion
    """

    R_D = Q(D)
    R_D_pub = Q(D_pub)
    distortion = np.abs(R_D - R_D_pub) / (
        max(R_D, R_D_pub) + 1e-12  # Avoid division by zero
    )
    return distortion


def query_distortion(D, D_pub, R, t_int, uncertainty):
    """
    Calculate the distortion of the query for the given dataset and published dataset.
    :param D: Original dataset
    :param D_pub: Published dataset
    :param bounds: Bounds for the dataset
    :param r: Radius for the query
    :return: Distortion values for psi and dai queries
    """

    q1 = lambda D: possibly_sometimes_inside(D, R, t_int, uncertainty)
    q2 = lambda D: definitely_always_inside(D, R, t_int, uncertainty)

    psi_distortion = range_query_distortion(D, D_pub, q1)
    dai_distortion = range_query_distortion(D, D_pub, q2)

    return psi_distortion, dai_distortion


def hausdorff(D, D_pub):
    """
    Compute the Hausdorff distance between two datasets.
    :param D: Original dataset
    :param D_pub: Published dataset
    :return: Hausdorff distance
    """
    D = D.reshape(D.shape[0], -1)
    D_pub = D_pub.reshape(D_pub.shape[0], -1)
    return max(
        directed_hausdorff(D, D_pub)[0],
        directed_hausdorff(D_pub, D)[0],
    )


def individual_hausdorff(D, D_pub):
    """
    Compute the "individual" Hausdorff distance for each T in D_pub
    .. math::
        IdvHD(T) = min_{T' in D} ||T - T'||_2
    """
    D = D.reshape(D.shape[0], -1)
    D_pub = D_pub.reshape(D_pub.shape[0], -1)
    return np.array([np.min(np.linalg.norm(D - T, axis=1)) for T in D_pub])
