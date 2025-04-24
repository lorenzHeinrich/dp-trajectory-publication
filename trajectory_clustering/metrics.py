import numpy as np

from scipy.spatial.distance import directed_hausdorff


def possibly_sometimes_inside(D, R, t_interval, uncertainties):
    """
    Compute the number of trajectories that are possibly inside a circular region
    at least once during the specified time interval, given pointwise uncertainty.

    Parameters:
        D: ndarray of shape (n, m, 2) — trajectory dataset
        R: ((x, y), r) — circular query region
        t_interval: (tb, te) — time interval (inclusive)
        uncertainties: ndarray of shape (n, m) — uncertainty radius for each point

    Returns:
        Number of trajectories that possibly intersect the region at least once
    """
    tb, te = t_interval
    (x, y), r = R

    D_slice = D[:, tb : te + 1, :]  # (n, t_span, 2)
    U_slice = uncertainties[:, tb : te + 1]  # (n, t_span)

    dx = D_slice[:, :, 0] - x
    dy = D_slice[:, :, 1] - y
    dist = np.sqrt(dx**2 + dy**2)

    inside = dist <= (r + U_slice)
    return np.any(inside, axis=1).sum()


def definitely_always_inside(D, R, t_interval, uncertainties):
    """
    Compute the number of trajectories that are definitely always inside a circular region,
    given per-point uncertainty.

    A point is considered definitely inside if its full uncertainty radius lies within the query circle.

    Parameters:
        D: ndarray of shape (n, m, 2) — trajectory dataset
        R: ((x, y), r) — circular query region
        t_interval: (tb, te) — time interval (inclusive)
        uncertainties: ndarray of shape (n, m) — uncertainty radius for each point

    Returns:
        Number of trajectories that are always fully within the region during the time interval
    """
    tb, te = t_interval
    (x, y), r = R

    D_slice = D[:, tb : te + 1, :]  # (n, t_span, 2)
    U_slice = uncertainties[:, tb : te + 1]  # (n, t_span)

    dx = D_slice[:, :, 0] - x
    dy = D_slice[:, :, 1] - y
    dist = np.sqrt(dx**2 + dy**2)

    inside = dist <= (r - U_slice)
    return np.all(inside, axis=1).sum()


def range_query_distortion(D, D_pub, U, Q):
    """
    Compute both absolute and relative range query distortion between
    the original and published datasets.

    Parameters:
        D: ndarray (n, t, 2) — Original dataset
        D_pub: ndarray (n, t, 2) — Sanitized dataset
        U: ndarray (n, t) — Per-point uncertainties for D_pub
        Q: callable — Query function of the form Q(D, U) -> float

    Returns:
        (abs_error, rel_error)
        abs_error: |Q(D) - Q(D_pub)|
        rel_error: abs_error / Q(D)
    """
    D_U = np.zeros(D.shape[:2])  # Original data has zero uncertainty
    R_D = Q(D, D_U)
    R_D_pub = Q(D_pub, U)

    abs_error = np.abs(R_D - R_D_pub)
    rel_error = abs_error / (R_D + 1e-12)

    return abs_error, rel_error


def query_distortion(D, D_pub, R, t_int, U):
    """
    Compute the query distortion for the given datasets and query region.
    :param D: Original dataset
    :param D_pub: Published dataset
    :param R: Query region (center, radius)
    :param t_int: Time interval (start, end)
    :param U: Uncertainty radii for each point of each trajectory
    """
    q1 = lambda D, U: possibly_sometimes_inside(D, R, t_int, U)
    q2 = lambda D, U: definitely_always_inside(D, R, t_int, U)

    psi_distortion = range_query_distortion(D, D_pub, U, q1)
    dai_distortion = range_query_distortion(D, D_pub, U, q2)

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
