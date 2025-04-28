import logging
import numpy as np
from diffprivlib.mechanisms import Laplace

from trajectory_clustering.dpapt.adaptive_cells import AdaptiveCells


logger = logging.getLogger(__name__)


class DPAPT:
    def __init__(
        self,
        ac=AdaptiveCells(),
        alpha=0.5,  # balance between grid and trajectory privacy
        thresh_traj=lambda eps: 2
        * np.sqrt(2)
        / eps,  # threshold for including a trajectory
        randomize=True,
    ):
        self.ac = ac
        self.alpha = alpha
        self.thresh_traj = thresh_traj
        self.randomize = randomize
        self.secure_random = False

    def publish(self, D, t_interval, bounds, eps):
        logger.debug(
            f"Starting dpapt with t_interval={t_interval}, eps={eps}, bounds={bounds}"
        )

        # Calculate privacy budget for the current time step
        tl, tu = t_interval
        eps_step = eps / (tu - tl + 1)
        eps_grid = eps_step * self.alpha
        eps_traj = eps_step * (1 - self.alpha)
        # call adaptive_cells to estimate the location domain for the current time step
        areas = self.ac.adaptive_cells(D[:, tu], bounds, eps_grid)
        counts = np.array([int(np.round(area.sum_counts())) for area in areas])
        if (tu - tl) == 0:
            trajects_new = np.array([np.array([area]) for area in areas])
            logger.debug(
                f"Returning {len(trajects_new)} trajectories, with counts summing to {np.sum(counts)}",
            )
            return trajects_new, counts

        # get all sanatized trajectories from the previous time step
        trajects_prev, _ = self.publish(D, (tl, tu - 1), bounds, eps=eps - eps_step)
        lap = Laplace(sensitivity=1, epsilon=eps_traj)

        logger.debug(
            f"Processing t_interval={t_interval} with {len(areas)} areas and {len(trajects_prev)} previous trajectories"
        )

        # initialize result arrays
        trajects_prev_len = len(trajects_prev)
        areas_len = len(areas)
        counts_len = areas_len * trajects_prev_len
        counts_true = np.empty(counts_len)  # true counts, tracked for debugging
        counts_rand = np.empty(counts_len)

        # preprocessing step that collects the trajectories in D that traverse the cell trajectories
        # this partitions the trajectories in D into disjoint sets for which the count queries are issued
        D_traj_prev = self._traverses(D, trajects_prev, t_interval)
        logger.debug(
            f"Traversing {len(D_traj_prev)} trajectories in D that traverse the area trajectories"
        )
        offsets = (
            np.arange(trajects_prev_len) * areas_len
        )  # the offset since we match each trajectory with all cells
        for offset, D_prev in zip(offsets, D_traj_prev):
            # obtain counts
            counts = self._counts_inside(D_prev, tu, areas)
            counts_true[offset : offset + areas_len] = counts

            # randomize counts
            if self.randomize:
                if self.secure_random:
                    counts = np.array([lap.randomise(count) for count in counts])
                else:
                    counts = counts + np.random.laplace(0, 1 / eps_traj, len(counts))
            counts_rand[offset : offset + areas_len] = counts

        logger.debug(
            f"We have preserved {np.sum(counts_true > 0)} trajectories with counts summing to {np.sum(counts_true)}"
        )

        # filter out the cells that have counts below the threshold
        thresh = self.thresh_traj(eps_traj) if self.randomize else 1
        valid_mask = counts_rand >= thresh
        counts_valid = counts_rand[valid_mask]
        counts_valid = np.array([int(np.round(count)) for count in counts_valid])

        # build the new trajectories based on the valid areas
        traj_idx, area_idx = np.where(valid_mask.reshape(trajects_prev_len, areas_len))
        trajects_new = np.array(
            [
                list(trajects_prev[traj_i]) + [areas[area_i]]
                for traj_i, area_i in zip(traj_idx, area_idx)
            ]
        )

        logger.debug(
            f"Returning {len(trajects_new)} trajectories, with counts summing to {np.sum(counts_valid)}",
        )

        return trajects_new, counts_valid

    def _counts_inside(self, D, t, areas):
        """
        Count the number of points in D that are inside each of the given areas.

        D: array of shape (n, m, 2)
        t: time index
        areas: list of Area objects
        Returns: array of shape (len(areas),)
        """
        n = D.shape[0]
        x, y = D[:, t, 0], D[:, t, 1]

        counts = np.zeros(len(areas), dtype=int)

        for i, area in enumerate(areas):
            mask = np.zeros(n, dtype=bool)
            for (xl, xu), (yl, yu) in area.cells:
                mask |= (x >= xl) & (x < xu) & (y >= yl) & (y < yu)
            counts[i] = np.sum(mask)

        return counts

    def _traverses(self, D, prev_trajects, t_int, batch_size=32):
        """
        Collect for each area-trajectory in prev_trajects the trajectories in D that traverse it.

        D: array of shape (n, m, 2)
        prev_trajects: list of k area-trajectories, each a list of m Area objects
        Returns: list of arrays of shape (n_i, m, 2), one for each matching area-trajectory
        """
        if len(prev_trajects) == 0:
            return []

        k = len(prev_trajects)
        traversing = []

        for batch_start in range(0, k, batch_size):
            batch = prev_trajects[batch_start : batch_start + batch_size]
            mask = self._traverses_batch(D, t_int, batch)
            for j in range(mask.shape[1]):
                traversing.append(D[mask[:, j]])

        return traversing

    def _traverses_batch(self, D, t_int, prev_trajects):
        """
        Check if trajectories in D are inside the area-trajectories in prev_trajects.

        D: array of shape (n, m, 2)
        t_int: (tl, tu) - the time interval
        prev_trajects: list of k area-trajectories, each a list of (tu - tl) Area objects
        Returns: boolean array of shape (n, k)
        """
        tl, tu = t_int
        n = D.shape[0]
        k = len(prev_trajects)

        result = np.ones((n, k), dtype=bool)

        for j, area_traj in enumerate(prev_trajects):
            for i, t in enumerate(range(tl, tu)):
                x = D[:, t, 0]
                y = D[:, t, 1]
                mask = np.zeros(n, dtype=bool)
                for (xl, xu), (yl, yu) in area_traj[i].cells:
                    mask |= (x >= xl) & (x < xu) & (y >= yl) & (y < yu)
                result[:, j] &= mask  # Only keep those that matched all timestamps

        return result
