import logging
from math import e
from time import time
import numpy as np
from diffprivlib.mechanisms import Laplace

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

thresh_default = lambda eps: 2 * np.sqrt(2) / eps


def dpapt(
    D,
    t_interval,
    bounds,
    eps,
    alpha,
    c1=10,
    thresh_grid=thresh_default,
    thresh_traj=thresh_default,
    randomize=True,
    secure_random=False,
):
    logging.info(
        f"Starting dpapt with t_interval={t_interval}, eps={eps}, alpha={alpha}"
    )
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
    cells, counts = _to_cells(grids, thresh_grid(eps_grid) if randomize else 1)

    if (t_u - t_l) == 0:
        logging.info("Reached base case of recursion")
        return np.array([np.array([cell]) for cell in cells]), counts

    trajects_prev, counts = dpapt(
        D,
        (t_l, t_u - 1),
        bounds,
        eps=eps - eps_step,
        alpha=alpha,
        c1=c1,
        randomize=randomize,
    )
    eps_traj = eps_step * (1 - alpha)
    lap = Laplace(sensitivity=1, epsilon=eps_traj)

    logging.info(
        f"Processing t_interval={t_interval} with {len(cells)} cells and {len(trajects_prev)} previous trajectories"
    )

    trajects_prev_len = len(trajects_prev)
    cells_len = len(cells)
    counts_len = cells_len * trajects_prev_len
    counts_true = np.empty(counts_len)
    counts_rand = np.empty(counts_len)

    start = time()
    D_traj_prev = traverses(D, trajects_prev)
    logging.info(f"Building lookup table took {time() - start:.2f}s")

    start = time()
    offsets = np.arange(trajects_prev_len) * cells_len
    for offset, D_prev in zip(offsets, D_traj_prev):
        counts = counts_inside(D_prev, t_u, cells)
        counts_true[offset : offset + cells_len] = counts
        if randomize:
            if secure_random:
                counts = np.array([lap.randomise(count) for count in counts])
            else:
                counts = counts + np.random.laplace(0, 1 / eps_traj, len(counts))
        counts_rand[offset : offset + cells_len] = counts
    logging.info(
        f"We have preserved {np.sum(counts_true > 0)} trajectories with counts summing to {np.sum(counts_true)}"
    )
    logging.info(f"Counting took {time() - start:.2f}s")

    start = time()
    thresh = thresh_traj(eps_traj) if randomize else 1
    valid_mask = counts_rand >= thresh
    counts_valid = counts_rand[valid_mask]

    num_new_traj = np.sum(valid_mask)
    trajects_new = np.empty((num_new_traj, t_u - t_l + 1, 2, 2))  # type: ignore

    traj_idx, cell_idx = np.where(valid_mask.reshape(trajects_prev_len, cells_len))

    for i, (traj_i, cell_i) in enumerate(zip(traj_idx, cell_idx)):
        trajects_new[i] = np.concatenate(
            (
                trajects_prev[traj_i],
                np.array([np.array(cells[cell_i])]),
            )
        )

    logging.info(f"Constructing result took {time() - start:.2f}s")

    logging.info(
        f"Returning {len(trajects_new)} trajectories, with counts summing to {np.sum(counts_valid)}",
    )

    return (np.array(trajects_new), np.array(counts_valid))


def inside_mask(D, t, cell):
    x, y = D[:, t, 0], D[:, t, 1]
    (x_l, x_u), (y_l, y_u) = cell
    mask = (x >= x_l) & (x < x_u) & (y >= y_l) & (y < y_u)
    return mask


def counts_inside(D, t, cells):
    x, y = D[:, t, 0], D[:, t, 1]
    x_l, x_u = cells[:, 0, 0], cells[:, 0, 1]
    y_l, y_u = cells[:, 1, 0], cells[:, 1, 1]

    inside = (
        (x[:, None] >= x_l)
        & (x[:, None] < x_u)
        & (y[:, None] >= y_l)
        & (y[:, None] < y_u)
    )

    return np.sum(inside, axis=0)


def traverses(D, prev_trajects, batch_size=32):
    if len(prev_trajects) == 0:
        return []
    batch_idx = np.arange(0, prev_trajects.shape[0], batch_size)
    traversing = []
    start = time()
    progress = 0
    checkpoint = max(
        1, len(prev_trajects) // 100
    )  # Sicherstellen, dass der Schritt mindestens 1 ist

    for i in range(len(batch_idx) - 1):
        mask = traverses_batch(D, prev_trajects[batch_idx[i] : batch_idx[i + 1]])
        traversing.extend(D[mask[:, i]] for i in range(mask.shape[1]))
        progress += batch_idx[i + 1] - batch_idx[i]

        if progress >= checkpoint:
            remaining_time = (
                (time() - start) / progress * (len(prev_trajects) - progress)
            )
            print(
                f"Progress: {progress}/{len(prev_trajects)} - Remaining time: {remaining_time:.2f}s",
                end="\r",
            )
            checkpoint += len(prev_trajects) // 100

    mask = traverses_batch(D, prev_trajects[batch_idx[-1] :])
    traversing.extend(D[mask[:, i]] for i in range(mask.shape[1]))
    return traversing


def traverses_batch(D, prev_trajects):
    D_truncated = D[:, None, : prev_trajects.shape[1], :]
    x, y = D_truncated[..., 0], D_truncated[..., 1]  # Coordinates of points in D
    x_l, x_u = (prev_trajects[..., 0, 0], prev_trajects[..., 0, 1])
    y_l, y_u = (prev_trajects[..., 1, 0], prev_trajects[..., 1, 1])

    in_cell = (x >= x_l) & (x < x_u) & (y >= y_l) & (y < y_u)

    return np.all(in_cell, axis=2)


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
    logging.info("Starting adaptive_noisy_grid")
    epsN = beta * eps
    epsl1 = alpha * (1 - beta) * eps if alpha is not None else (1 - beta) * eps
    epsl2 = (1 - alpha) * (1 - beta) * eps if alpha is not None else epsl1

    lap = Laplace(sensitivity=1, epsilon=epsN)
    N = max(
        1,
        Laplace(sensitivity=1, epsilon=epsN).randomise(len(L)) if randomize else len(L),
    )
    m1 = max(10, int(np.ceil(1 / 4 * np.ceil(np.sqrt(N * eps / c1)))))
    print(f"m1: {m1}, N: {N}")
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

    logging.info("Finished adaptive_noisy_grid")
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


def post_process_uniform(D_cells):
    logging.info("Starting post_process_uniform")
    D_randomized = np.empty(
        (*D_cells.shape[0:2], 2), dtype=object
    )  # Preallocate output array

    for i, traj in enumerate(D_cells):
        random_traj = np.array(
            [
                (
                    np.random.uniform(x_l, x_u),
                    np.random.uniform(y_l, y_u),
                )  # Sample random point inside cell
                for ((x_l, x_u), (y_l, y_u)) in traj
            ]
        )
        D_randomized[i] = random_traj  # Store new trajectory

    logging.info("Finished post_process_uniform")
    return np.array(D_randomized, dtype=float)  # Convert to float array


def post_process_centroid(D_cells):
    logging.info("Starting post_process_centroid")
    D_centroids = np.empty(
        (*D_cells.shape[0:2], 2), dtype=object
    )  # Preallocate output array

    for i, traj in enumerate(D_cells):
        centroid_traj = np.array(
            [
                (
                    (x_l + x_u) / 2,
                    (y_l + y_u) / 2,
                )  # Sample random point inside cell
                for ((x_l, x_u), (y_l, y_u)) in traj
            ]
        )
        D_centroids[i] = centroid_traj  # Store new trajectory

    logging.info("Finished post_process_centroid")
    return np.array(D_centroids, dtype=float)  # Convert to float array
