import logging
import numpy as np
from diffprivlib.mechanisms import Laplace

from trajectory_clustering.dpapt.adaptive_cells import AdaptiveCells

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

thresh_default = lambda eps: 2 * np.sqrt(2) / eps


class DPAPT:
    def __init__(
        self,
        ac=AdaptiveCells(),
        alpha=0.5,  # balance between grid and trajectory privacy
        thresh_traj=thresh_default,  # threshold for including a trajectory
        randomize=True,
    ):
        self.ac = ac
        self.alpha = alpha
        self.thresh_traj = thresh_traj
        self.randomize = randomize
        self.secure_random = False

    def publish(self, D, t_interval, bounds, eps):
        logger.info(
            f"Starting dpapt with t_interval={t_interval}, eps={eps}, bounds={bounds}"
        )

        # Calculate privacy budget for the current time step
        tl, tu = t_interval
        eps_step = eps / (tu - tl + 1)
        eps_grid = eps_step * self.alpha
        eps_traj = eps_step * (1 - self.alpha)

        # call adaptive_cells to estimate the location domain for the current time step
        cells, counts = self.ac.adaptive_cells(D[:, tu], bounds, eps_grid)  # type: ignore

        if (tu - tl) == 0:
            trajects_new = np.array([np.array([cell]) for cell in cells])
            logger.debug(
                f"Returning {len(trajects_new)} trajectories, with counts summing to {np.sum(counts)}",
            )
            return trajects_new, counts

        # get all sanatized trajectories from the previous time step
        trajects_prev, _ = self.publish(D, (tl, tu - 1), bounds, eps=eps - eps_step)
        lap = Laplace(sensitivity=1, epsilon=eps_traj)

        logger.debug(
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
        logger.debug(
            f"Traversing {len(D_traj_prev)} trajectories in D that traverse the cell trajectories"
        )
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

        logger.info(
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

        logger.info(
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


def post_process_uniform(D_cells):
    logger.info("Starting post_process_uniform")
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

    logger.info("Finished post_process_uniform")
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
