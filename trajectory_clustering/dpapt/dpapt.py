import logging
import numpy as np
from diffprivlib.mechanisms import Laplace

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

thresh_default = lambda eps: 2 * np.sqrt(2) / eps


class DPAPT:
    def __init__(
        self,
        alpha=0.5,  # balance between grid and trajectory privacy
        beta=0.5,  # balance between l1 and l2 privacy
        gamma=0.1,  # balance between size estimation and grid privacy
        c=10,
        thresh_grid=thresh_default,
        thresh_traj=thresh_default,
        randomize=True,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c = c
        self.thresh_grid = thresh_grid
        self.thresh_traj = thresh_traj
        self.randomize = randomize
        self.secure_random = False

    def publish(self, D, t_interval, bounds, eps):
        logging.info(
            f"Starting dpapt with t_interval={t_interval}, eps={eps}, bounds={bounds}"
        )

        # Calculate privacy budget for the current time step
        tl, tu = t_interval
        eps_step = eps / (tu - tl + 1)
        eps_grid = eps_step * self.alpha
        eps_traj = eps_step * (1 - self.alpha)

        # call adaptive_cells to estimate the location domain for the current time step
        cells, counts = self._adaptive_cells(D[:, tu], bounds, eps_grid)

        if (tu - tl) == 0:
            return np.array([np.array([cell]) for cell in cells]), counts

        # get all sanatized trajectories from the previous time step
        trajects_prev, _ = self.publish(D, (tl, tu - 1), bounds, eps=eps - eps_step)
        lap = Laplace(sensitivity=1, epsilon=eps_traj)

        logging.info(
            f"Processing t_interval={t_interval} with {len(cells)} cells and {len(trajects_prev)} previous trajectories"
        )

        # initialize result arrays
        trajects_prev_len = len(trajects_prev)
        cells_len = len(cells)
        counts_len = cells_len * trajects_prev_len
        counts_true = np.empty(counts_len)  # true counts, tracked for debugging
        counts_rand = np.empty(counts_len)

        # preprocessing step that collects the trajectories in D that traverse the cell trajectories
        # this partitions the trajectories in D into disjoint sets for which the count queries are issued
        D_traj_prev = self._traverses(D, trajects_prev)

        offsets = (
            np.arange(trajects_prev_len) * cells_len
        )  # the offset since we match each trajectory with all cells
        for offset, D_prev in zip(offsets, D_traj_prev):
            # obtain counts
            counts = self._counts_inside(D_prev, tu, cells)
            counts_true[offset : offset + cells_len] = counts

            # randomize counts
            if self.randomize:
                if self.secure_random:
                    counts = np.array([lap.randomise(count) for count in counts])
                else:
                    counts = counts + np.random.laplace(0, 1 / eps_traj, len(counts))
            counts_rand[offset : offset + cells_len] = counts

        logging.info(
            f"We have preserved {np.sum(counts_true > 0)} trajectories with counts summing to {np.sum(counts_true)}"
        )

        # filter out the cells that have counts below the threshold
        thresh = self.thresh_traj(eps_traj) if self.randomize else 1
        valid_mask = counts_rand >= thresh
        counts_valid = counts_rand[valid_mask]
        counts_valid = np.array([int(np.round(count)) for count in counts_valid])

        # build the new trajectories based on the valid cells
        num_new_traj = int(np.sum(valid_mask))
        trajects_new = np.empty((num_new_traj, tu - tl + 1, 2, 2))

        traj_idx, cell_idx = np.where(valid_mask.reshape(trajects_prev_len, cells_len))
        for i, (traj_i, cell_i) in enumerate(zip(traj_idx, cell_idx)):
            trajects_new[i] = np.concatenate(
                (
                    trajects_prev[traj_i],
                    np.array([np.array(cells[cell_i])]),
                )
            )

        logging.info(
            f"Returning {len(trajects_new)} trajectories, with counts summing to {np.sum(counts_valid)}",
        )

        return trajects_new, counts_valid

    def _counts_inside(self, D, t, cells):
        """
        Count the number of points in D that are inside the cells.
        D: array of shape (n, m, 2)
        t: time index
        cells: a single cells trajectory - array of shape (k, 2, 2)
        Returns: array of shape (k,)
        """
        if len(cells) == 0:
            return np.zeros(0)
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

    def _traverses(self, D, prev_trajects, batch_size=32):
        """
        Collect for each cell trajectory in prev_trajects the trajectories in D that traverse it.
        D: array of shape (n, m, 2)
        prev_trajects: array of shape (k, m, 2, 2)
        Returns: list of arrays of shape (n, m, 2) the same length as prev_trajects
        """
        if len(prev_trajects) == 0:
            return []
        batch_idx = np.arange(0, prev_trajects.shape[0], batch_size)
        traversing = []

        for i in range(len(batch_idx) - 1):
            mask = self._traverses_batch(
                D, prev_trajects[batch_idx[i] : batch_idx[i + 1]]
            )
            traversing.extend(D[mask[:, j]] for j in range(mask.shape[1]))

        mask = self._traverses_batch(D, prev_trajects[batch_idx[-1] :])
        traversing.extend(D[mask[:, i]] for i in range(mask.shape[1]))
        return traversing

    def _traverses_batch(self, D, prev_trajects):
        """
        Check if trajectories in D are inside the cell trajectories in prev_trajects.

        D: array of shape (n, m, 2)
        prev_trajects: array of shape (k, m, 2, 2)
        Returns: boolean array of shape (n, k)
        """
        D_truncated = D[:, : prev_trajects.shape[1]]  # (n, m, 2)

        x = D_truncated[..., 0][:, None, :]  # (n, 1, m)
        y = D_truncated[..., 1][:, None, :]  # (n, 1, m)

        x_l = prev_trajects[..., 0, 0][None, :, :]  # (1, k, m)
        x_u = prev_trajects[..., 0, 1][None, :, :]  # (1, k, m)
        y_l = prev_trajects[..., 1, 0][None, :, :]  # (1, k, m)
        y_u = prev_trajects[..., 1, 1][None, :, :]  # (1, k, m)

        in_cell = (x >= x_l) & (x < x_u) & (y >= y_l) & (y < y_u)  # (n, k, m)

        return np.all(in_cell, axis=2)  # (n, k)

    def _adaptive_cells(self, L, bounds, eps):
        epsN = self.gamma * eps
        eps_remaining = eps - epsN
        epsl1 = self.beta * eps_remaining
        epsl2 = (1 - self.beta) * eps_remaining

        lap = Laplace(sensitivity=1, epsilon=epsN)
        N = max(
            1,
            (
                Laplace(sensitivity=1, epsilon=epsN).randomise(len(L))
                if self.randomize
                else len(L)
            ),
        )
        m1 = max(10, int(np.ceil(1 / 4 * np.ceil(np.sqrt(N * eps / self.c)))))

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
            if self.randomize
            else l1_grid
        )

        l2_grids = self._build_l2_grids(l1_grid, epsl2, xl1_step, yl1_step, x_l, y_l)

        l2_grids = self._obtain_l2_counts(
            L, l1_grid, l2_grids, x_l, y_l, xl1_step, yl1_step, epsl2
        )
        cells, counts = self._to_cells(
            l2_grids, self.thresh_grid(eps) if self.randomize else 1
        )
        return cells, counts

    def _build_l2_grids(self, l1_grid, epsl2, xl1_step, yl1_step, x_l, y_l):
        c2 = self.c / 2
        l2_grids = {}
        for i, j in np.ndindex(l1_grid.shape):
            nc = l1_grid[i, j]
            m2 = max(1, int(np.ceil(np.sqrt(max(0, nc) * epsl2 / c2))))
            l2_grids[(i, j)] = (
                m2,
                xl1_step / m2,
                yl1_step / m2,
                (i * xl1_step + x_l, j * yl1_step + y_l),
                np.zeros((m2, m2)),
            )
        return l2_grids

    def _obtain_l2_counts(
        self, L, l1_grid, l2_grids, x_l, y_l, xl1_step, yl1_step, epsl2
    ):
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
            if self.randomize
            else l2_grids
        )

        self._apply_constraint_inference(l1_grid, l2_grids)
        return l2_grids

    def _apply_constraint_inference(self, l1_grid, l2_grids):
        for i, j in np.ndindex(l1_grid.shape):
            l2_counts = l2_grids[(i, j)][4]
            leafs_nc_sum = np.sum(l2_counts)
            m2 = l2_grids[(i, j)][4].shape[0]
            root_nc = l1_grid[i, j]

            normalization = m2**2 * self.beta**2 + (1 - self.beta) ** 2
            root_nc_weight = self.beta**2 * m2**2 / normalization
            leafs_sum_weight = (1 - self.beta) ** 2 / normalization
            nc_wavg = root_nc_weight * root_nc + leafs_sum_weight * leafs_nc_sum

            l1_grid[i, j] = nc_wavg
            leaf_diff = (nc_wavg - leafs_nc_sum) / m2
            for x, y in np.ndindex(l2_counts.shape):
                l2_grids[(i, j)][4][x, y] += leaf_diff

    def _to_cells(self, l2_grids, thresh):
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
                )  # Compute centroid of cell
                for ((x_l, x_u), (y_l, y_u)) in traj
            ]
        )
        D_centroids[i] = centroid_traj  # Store new trajectory

    logging.info("Finished post_process_centroid")
    return np.array(D_centroids, dtype=float)  # Convert to float array
