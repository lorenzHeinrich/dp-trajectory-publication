from time import time
import numpy as np
from diffprivlib.mechanisms import Laplace


def dpapt(D, t_interval, bounds, eps, alpha, c1=10, randomize=True):
    t_l, t_u = t_interval
    eps_step = eps / (t_u - t_l + 1)
    eps_grid = eps_step * alpha
    grids = adaptive_noisy_grid(
        D[:, t_u],
        bounds,
        eps_grid,
        randomize,
        c1=c1,
        alpha=0.5 if t_u - t_l == 0 else None,
    )
    cells, counts = _to_cells(grids, 1 / eps_grid if randomize else 1)

    if (t_u - t_l) == 0:
        return np.array([np.array([cell]) for cell in cells]), counts

    trajects_prev, counts = dpapt(
        D, (t_l, t_u - 1), bounds, eps=eps - eps_step, alpha=alpha, randomize=randomize
    )

    eps_traj = eps_step * (1 - alpha)
    thresh = 2 * np.sqrt(2) / eps_traj if randomize else 1
    lap = Laplace(sensitivity=1, epsilon=eps_traj)

    trajects_new = np.ndarray((0, t_u - t_l + 1, *trajects_prev.shape[2:]))
    counts_new = np.array([])

    for cell in cells:
        trajects_cell = np.ndarray((0, t_u - t_l + 1, *cell.shape))
        cell_counts = np.array([])
        D_cell = inside(D, t_u, cell)

        for tr_prev in trajects_prev:
            tr = np.concatenate((tr_prev, [cell]))
            count = _count_traversing(D_cell, tr)
            nc = lap.randomise(count) if randomize else count

            if nc >= thresh:
                trajects_cell = np.concatenate((trajects_cell, [tr]))
                cell_counts = np.concatenate((cell_counts, [nc]))

        trajects_new = np.concatenate((trajects_new, trajects_cell))
        counts_new = np.concatenate((counts_new, cell_counts))

    return trajects_new, counts_new


def inside(D, t, cell):
    return np.array([tr for tr in D if is_in(tr[t], cell)])


def is_in(l, cell):
    (x, y) = l
    ((x_l, x_u), (y_l, y_u)) = cell
    return x_l <= x < x_u and y_l <= y < y_u


def _count_traversing(D, cells_tr):

    def traverses(tr, cells_tr):
        return np.all(
            [is_in(l, cell) for l, cell in zip(tr[: len(cells_tr)], cells_tr)]
        )

    return np.sum([1 for tr in D if traverses(tr, cells_tr)])


def _to_cells(l2_grids, thresh):
    cells = []
    counts = []
    for m2, x_step, y_step, (x_start, y_start), cell_counts in l2_grids.values():
        if np.ndim(cell_counts) == 0 and cell_counts < thresh:
            continue
        for i, j in np.ndindex((m2, m2)):
            x_span = (x_start + i * x_step, x_start + (i + 1) * x_step)
            y_span = (y_start + j * y_step, y_start + (j + 1) * y_step)
            count = cell_counts if np.ndim(cell_counts) == 0 else cell_counts[i, j]
            if count >= thresh:
                cells.append((x_span, y_span))
                counts.append(count)
    return np.array(cells), np.array(counts)


def adaptive_noisy_grid(L, bounds, eps, randomize, c1=10, beta=0.1, alpha=None):
    epsN = beta * eps
    epsl1 = alpha * (1 - beta) * eps if alpha is not None else (1 - beta) * eps
    epsl2 = (1 - alpha) * (1 - beta) * eps if alpha is not None else epsl1

    lap = Laplace(sensitivity=1, epsilon=epsN)
    N = max(
        1,
        Laplace(sensitivity=1, epsilon=epsN).randomise(len(L)) if randomize else len(L),
    )
    m1 = max(4, int(np.ceil(1 / 4 * np.ceil(np.sqrt(N * eps / c1)))))
    ((x_l, x_u), (y_l, y_u)) = bounds
    xl1_step = (x_u - x_l) / m1
    yl1_step = (y_u - y_l) / m1

    lap = Laplace(sensitivity=1, epsilon=epsl1)
    l1_grid = np.zeros((m1, m1))
    for x, y in L:
        i = int((x - x_l) / xl1_step)
        j = int((y - y_l) / yl1_step)
        l1_grid[i, j] += 1
    l1_grid = (
        np.array([[lap.randomise(n) for n in row] for row in l1_grid])
        if randomize
        else l1_grid
    )

    l2_grids = _build_l2_grids(
        l1_grid, epsl2, c1, xl1_step, yl1_step, x_l, y_l, alpha != None
    )

    if alpha is not None:
        for x, y in L:
            i = int((x - x_l) / xl1_step)
            j = int((y - y_l) / yl1_step)
            _, xl2_step, yl2_step, _, l2_grid = l2_grids[(i, j)]
            x2 = int((x - (x_l + i * xl1_step)) / xl2_step)
            y2 = int((y - (y_l + j * yl1_step)) / yl2_step)
            l2_grid[x2, y2] += 1
        lap = Laplace(sensitivity=1, epsilon=epsl2)
        l2_grids = (
            {
                k: (
                    m2,
                    x_step,
                    y_step,
                    loc,
                    np.array(
                        [[lap.randomise(count) for count in row] for row in counts]
                    ),
                )
                for k, (m2, x_step, y_step, loc, counts) in l2_grids.items()
            }
            if randomize
            else l2_grids
        )

        _apply_constraint_inference(l1_grid, l2_grids, alpha)

    return l2_grids


def _build_l2_grids(l1_grid, epsl2, c1, xl1_step, yl1_step, x_l, y_l, for_counts=False):
    c2 = c1 / 2
    l2_grids = {}
    for i, j in np.ndindex(l1_grid.shape):
        nc = l1_grid[i, j]
        m2 = max(1, int(np.ceil(np.sqrt(max(0, nc) * epsl2 / c2))))
        l2_grids[(i, j)] = (
            m2,
            xl1_step / m2,
            yl1_step / m2,
            (i * xl1_step + x_l, j * yl1_step + y_l),
            np.zeros((m2, m2)) if for_counts else nc,
        )
    return l2_grids


def _apply_constraint_inference(l1_grid, l2_grids, alpha):
    for i, j in np.ndindex(l1_grid.shape):
        l2_counts = l2_grids[(i, j)][4]
        leafs_nc_sum = np.sum(l2_counts)
        m2 = l2_grids[(i, j)][4].shape[0]
        root_nc = l1_grid[i, j]

        normalization = m2**2 * alpha**2 + (1 - alpha) ** 2
        root_nc_weight = alpha**2 * m2**2 / normalization
        leafs_sum_weight = (1 - alpha) ** 2 / normalization
        nc_wavg = root_nc_weight * root_nc + leafs_sum_weight * leafs_nc_sum

        l1_grid[i, j] = nc_wavg
        leaf_diff = (nc_wavg - leafs_nc_sum) / m2
        for x, y in np.ndindex(l2_counts.shape):
            l2_grids[(i, j)][4][x, y] += leaf_diff
