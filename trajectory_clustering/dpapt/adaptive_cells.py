import logging

import numpy as np

from diffprivlib.mechanisms import Laplace
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)


class AdaptiveCells:
    def __init__(
        self,
        c=10.0,
        f_m1=lambda N, eps, c: max(10, 1 / 4 * np.ceil(np.ceil(np.sqrt(N * eps / c)))),
        f_m2=lambda nc, eps, c: int(np.ceil(np.sqrt(max(1, nc * eps) / (c / 2)))),
        beta=0.5,  # balance between l1 and l2 privacy
        gamma=0.1,  # balance between size estimation and grid privacy
        thresh_grid=lambda eps: 2 * np.sqrt(2) / eps,
        do_filter=True,
        n_clusters=None,
        randomize=True,
    ):
        self.c = c
        self.f_m1 = f_m1
        self.f_m2 = f_m2
        self.beta = beta
        self.gamma = gamma
        self.thresh_grid = thresh_grid
        self.do_filter = do_filter
        self.n_clusters = n_clusters
        self.randomize = randomize

    def adaptive_cells(self, L, bounds, eps):
        epsN = self.gamma * eps
        eps_remaining = eps - epsN
        epsl1 = self.beta * eps_remaining
        epsl2 = (1 - self.beta) * eps_remaining

        lap = Laplace(sensitivity=1, epsilon=epsN)
        N = max(1, lap.randomise(len(L)) if self.randomize else len(L))

        logger.debug(
            f"Estimated N={N} with epsN={epsN} = {self.gamma} * {eps} - Error: {np.abs(N-len(L)) / len(L)}"
        )
        m1 = self.f_m1(N, eps, self.c)
        logger.debug(f"m1={m1}")

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
        cells, counts = self._to_cells(l2_grids, eps)

        if self.n_clusters != None:
            labels, centers = AGkM(cells, counts, self.n_clusters)
            return cells, counts, labels, centers

        return cells, counts

    def _build_l2_grids(self, l1_grid, epsl2, xl1_step, yl1_step, x_l, y_l):
        l2_grids = {}
        for i, j in np.ndindex(l1_grid.shape):
            nc = l1_grid[i, j]
            m2 = self.f_m2(nc, epsl2, self.c)

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

    def _to_cells(self, l2_grids, eps):
        cells = []
        counts = []
        thresh = self.thresh_grid(eps) if self.randomize and self.do_filter else 1
        for m2, x_step, y_step, (x_start, y_start), cell_counts in l2_grids.values():
            for i, j in np.ndindex((m2, m2)):
                x_span = (x_start + i * x_step, x_start + (i + 1) * x_step)
                y_span = (y_start + j * y_step, y_start + (j + 1) * y_step)
                count = cell_counts[i, j]
                if not self.do_filter or (count >= thresh):
                    cells.append((x_span, y_span))
                    counts.append(count)
        return np.array(cells), np.array(counts)


def AGkM(cells, counts, k):
    X = np.array([[(xl + xu) / 2, (yl + yu) / 2] for (xl, xu), (yl, yu) in cells])
    props = np.abs(counts) / np.sum(np.abs(counts))
    C = X[np.random.choice(X.shape[0], k, replace=False, p=props)]

    km = KMeans(n_clusters=k, init=C).fit(X, sample_weight=counts)
    return km.labels_, km.cluster_centers_
