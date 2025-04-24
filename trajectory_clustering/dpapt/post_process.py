import numpy as np
from trajectory_clustering.dpapt.adaptive_cells import cell_to_center


def post_process_with_uncertainty(D_areas, counts, sample=False, uniform=False):
    """
    Convert a trajectory dataset of Area sequences to point trajectories and derive uncertainties.

    Parameters:
        D_areas: array-like of shape (n, m), each element is an Area
        counts: array-like of length n, number of times to replicate each trajectory
        sample: if True, sample a cell from each area; else use area center or closest cell
        uniform: if True, sample a point inside the selected cell; else use its center

    Returns:
        D_out: np.ndarray of shape (sum(counts), m, 2) — point trajectories
        U_out: np.ndarray of shape (sum(counts), m)    — uncertainty radii
    """
    total = np.sum(counts)
    m = D_areas.shape[1]

    D_out = np.empty((total, m, 2))
    U_out = np.empty((total, m))

    idx = 0
    for area_traj, count in zip(D_areas, counts):
        for _ in range(count):
            traj_points = []
            traj_uncertainties = []

            for area in area_traj:
                point, radius = derive_point_and_uncertainty(area, sample, uniform)
                traj_points.append(point)
                traj_uncertainties.append(radius)

            D_out[idx] = traj_points
            U_out[idx] = traj_uncertainties
            idx += 1

    return D_out, U_out


def derive_point_and_uncertainty(area, sample, uniform):
    match (sample, uniform):
        case (False, False):
            # Case 1: use area center, uncertainty = weighted area radius
            point = area.center
            radius = area_weighted_radius(area)

        case (False, True):
            # Case 2: uniform point from closest cell, uncertainty = weighted area radius
            cell = area.cell_closest_to_center()
            (xl, xu), (yl, yu) = cell
            point = (
                np.random.uniform(xl, xu),
                np.random.uniform(yl, yu),
            )
            radius = area_weighted_radius(area)

        case (True, False):
            # Case 3: sampled cell center, uncertainty = radius of that cell
            cell = area.select_cell()
            point = cell_to_center(cell)
            radius = cell_radius(cell)

        case (True, True):
            # Case 4: uniform point from sampled cell, uncertainty = 0
            cell = area.select_cell()
            (xl, xu), (yl, yu) = cell
            point = (
                np.random.uniform(xl, xu),
                np.random.uniform(yl, yu),
            )
            radius = 0.0
        case _:
            raise ValueError("Invalid combination of sample and uniform flags")
    return point, radius


def cell_radius(cell):
    """
    Calculate the radius of a cell.

    Parameters:
        cell: tuple of two tuples, each containing two floats (xl, xu), (yl, yu)

    Returns:
        float: radius of the cell
    """
    (xl, xu), (yl, yu) = cell
    return np.sqrt((xu - xl) ** 2 + (yu - yl) ** 2) / 2


def area_weighted_radius(area):
    """
    Calculate the weighted average distance of the cells to the area center.
    For singleton areas (1 cell), return the radius of that cell.

    Parameters:
        area: Area object
    Returns:
        float: weighted average distance of the cells to the area center
    """
    if len(area.cells) == 1:
        return cell_radius(area.cells[0])

    centers = np.array([cell_to_center(cell) for cell in area.cells])
    distances = np.linalg.norm(centers - area.center, axis=1)
    counts = np.maximum(area.counts, 0)

    return np.average(distances, weights=counts)
