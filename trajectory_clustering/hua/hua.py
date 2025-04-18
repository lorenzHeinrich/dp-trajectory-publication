import logging
from math import log
import secrets
import numpy as np
from numpy import float64, floating, int64
from sklearn.cluster import KMeans
from diffprivlib.mechanisms import Exponential, Laplace

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Modification:
    def __init__(
        self,
        id: int,
        from_cluster: int | int64,
        to_cluster: int | int64,
        distance: float | floating,
    ) -> None:
        self.id = id
        self.from_cluster = int64(from_cluster)
        self.to_cluster = int64(to_cluster)
        self.distance = float64(distance)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Modification):
            return False
        return (
            self.id == value.id
            and self.from_cluster == value.from_cluster
            and self.to_cluster == value.to_cluster
            and self.distance == value.distance
        )

    def __repr__(self) -> str:
        return f"Modification({self.id!r}, {self.from_cluster!r}, {self.to_cluster!r} {self.distance!r})"


class Hua:
    def __init__(self, m, phi, eps):
        self.m = m
        self.phi = phi
        self.eps1 = eps / 2
        self.eps2 = eps - self.eps1

    def publish(self, D: np.ndarray):
        logger.info(
            "Starting Hua algorithm with m=%d, phi=%d, eps=%.2f",
            self.m,
            self.phi,
            self.eps1,
        )
        D = D.reshape(D.shape[0], -1)
        generalized = self._dp_location_generalization(D)
        trajects, counts = self._dp_release(D, generalized)

        return trajects.reshape((trajects.shape[0], -1, 2)), counts

    def _dp_location_generalization(self, D: np.ndarray):
        logger.info("Applying k-means clustering")
        kmeans = KMeans(n_clusters=self.m).fit(D)
        logger.info("Constuctiong sub-optimal partitions")
        p_opt = (kmeans.labels_, kmeans.cluster_centers_)
        modifications = self._phi_sub_optimal(D, p_opt)
        s_partitions = self._s_kmeans_partitions(D)

        p_opt_modifications = self.apply_modifications(D, p_opt, modifications)

        partitions = p_opt_modifications + [(p_opt, lambda D: D)] + s_partitions

        p_opt_mean_dist = self._mean_distance(D, p_opt)
        utility_score = lambda x: p_opt_mean_dist / self._mean_distance(
            x[1](D), x[0]
        )  # x.mean_distance() may be zero, but highly unlikely for real-world data
        logger.info("Calculating utility scores for %d partitions", len(partitions))
        utilities = [utility_score(p) for p in partitions]
        exp = Exponential(
            epsilon=self.eps1, sensitivity=1, utility=utilities, candidates=partitions
        )
        ((_, centers), _) = exp.randomise()  # type: ignore
        logger.info("Returning generalized trajectories")
        universes = self._location_universes(centers)
        return universes

    def _dp_release(self, D: np.ndarray, universes: np.ndarray):
        logger.info("Obtaining noisy counts for generalized trajectories")
        # filter out non-positive counted trajectories and sort by count
        generalized, noisy_counts = self._make_noisy_counts(D, universes)
        nz_mask = noisy_counts > 0
        generalized = generalized[nz_mask]
        noisy_counts = noisy_counts[nz_mask]
        sorted_pairs = sorted(
            zip(generalized, noisy_counts), key=lambda x: x[1], reverse=True
        )
        generalized, noisy_counts = map(np.array, zip(*sorted_pairs))

        size_omega = np.prod([u.shape[0] for u in universes], dtype=object)
        size_remaining_omega = size_omega - generalized.shape[0]
        total_count = 0
        release_trajects = np.empty((0, generalized.shape[1]))
        release_counts = []
        logger.info(
            "Doing noisy count estimation for %d trajectories", size_remaining_omega
        )
        for i, (ci, cj) in enumerate(zip(noisy_counts[:-1], noisy_counts[1:])):
            # f(x, b) = 1/(2b) e^(-x/b)
            # using b = ε:
            # f(x, ε) = 1/(2ε) e^(-x/ε)
            # ∫ 1/(2ε) e^(-x/ε) dx = -1/2 e^(-x/ε)
            # using b = 1/ε:
            # f(x, 1/ε) = 1/2 ε e^(-xε)
            # ∫ 1/2 ε e^(-x ε) dx = -1/2 e^(-xε)
            antiderivative = lambda x: -1 / 2 * np.exp(-x / self.eps2)
            integral = lambda a, b: antiderivative(b) - antiderivative(a)

            # num_i = |Ω - D'| * ∫_{c_j}^{c_i} f(x, ε)
            num_i = int(np.round(size_remaining_omega * integral(cj, ci)))

            max_num = min(num_i, int(D.shape[0] - total_count))
            logger.info(
                "Generating min(%d, %d) random trajectories for noisy count interval (%d, %d]",
                num_i,
                D.shape[0] - total_count,
                cj,
                ci,
            )
            if num_i > 0:
                rand_counts = np.array(
                    [secrets.randbelow(int(ci - cj)) + cj for _ in range(max_num)]
                )
                rand_trajects = np.array(
                    [self._draw_trajectory(universes) for _ in range(max_num)]
                )
                cumulative_counts = np.cumsum(rand_counts)
                valid_indices = cumulative_counts + total_count <= D.shape[0]
                valid_rand_trajects = rand_trajects[valid_indices]
                valid_rand_counts = rand_counts[valid_indices]
                release_trajects = np.concatenate(
                    (release_trajects, valid_rand_trajects)
                )
                release_counts.extend(valid_rand_counts.tolist())
                total_count += (
                    cumulative_counts[valid_indices][-1] if valid_indices.any() else 0
                )

            release_trajects = np.concatenate((release_trajects, [generalized[i]]))
            release_counts.append(noisy_counts[i])
            total_count += noisy_counts[i]

            if total_count >= D.shape[0]:
                break

        return release_trajects, release_counts

    def _phi_sub_optimal(self, D: np.ndarray, p_opt) -> list[list[Modification]]:
        candidate_mods = self._phi_sub_optimal_individual(D, p_opt)
        best_sets: list[list[Modification]] = [[candidate_mods[0]]]

        logger.info(
            "Calculating phi sub-optimal modifications for %d candidates",
            len(candidate_mods),
        )
        for mod in candidate_mods[1:]:
            new_sets: list[list[Modification]] = []

            for mod_set in best_sets:
                used_ids = {m.id for m in mod_set}
                if mod.id not in used_ids:
                    new_set = mod_set + [mod]
                    new_sets.append(new_set)

            best_sets += new_sets
            # Sort sets by total utility loss
            best_sets.sort(key=lambda mods: sum(float(m.distance) for m in mods))

            # Keep only top φ sets
            if len(best_sets) > self.phi:
                best_sets = best_sets[: self.phi]

        return best_sets

    def _phi_sub_optimal_individual(self, D: np.ndarray, p_opt) -> list[Modification]:
        logger.info("Calculating phi sub-optimal individual modifications")
        labels, centers = p_opt
        candidate_mods: list[Modification] = []

        for i, traj in enumerate(D):
            current_cluster = labels[i]
            original_dist = np.linalg.norm(traj - centers[current_cluster])

            for k in set(labels) - {current_cluster}:
                new_dist = np.linalg.norm(traj - centers[k])
                loss = new_dist - original_dist
                candidate_mods.append(Modification(i, current_cluster, k, loss))

        # Select top-ϕ modifications by lowest utility loss
        return sorted(candidate_mods, key=lambda mod: float(mod.distance))[: self.phi]

    def _s_kmeans_partitions(self, D: np.ndarray):
        logger.info("Calculating %d-k-means partitions", D.shape[0])
        partitions = []
        for id in range(D.shape[0]):
            D_ = np.delete(D, id, axis=0)
            p = KMeans(n_clusters=self.m).fit(D_)
            partitions.append(
                ((p.labels_, p.cluster_centers_), lambda D: np.delete(D, id, axis=0))
            )
        return partitions

    def _mean_distance(self, D: np.ndarray, p) -> float:
        # MeanDist(p) = 1 / (m * |D_{LS_k^p}|) * sum_{k=1}^m sum_{traj in D_{LS_k^p}} || traj - c_k^p ||
        # where c_k^p is the mean trajectory of cluster k in partition p
        # and D_{LS_k^p} is the set of trajectories in cluster k in partition p
        labels, centers = p
        m = len(centers)
        sum = 0
        for k in range(m):
            group = D[labels == k]
            if len(group) > 0:
                group_sum = np.sum(np.linalg.norm(group - centers[k], axis=1))
                sum += group_sum / len(group)
        return float(sum / m)

    def apply_modifications(self, D: np.ndarray, p_opt, mods: list[list[Modification]]):
        partitions = []
        opt_labels, opt_centers = p_opt
        for mod in mods:
            labels = opt_labels.copy()
            centers = opt_centers.copy()
            for m in mod:
                labels[m.id] = m.to_cluster

            # recalculate mean - not clear from the paper
            centers = np.array(
                [
                    (
                        np.mean(D[labels == i], axis=0)
                        if len(D[labels == i]) > 0
                        else np.zeros(D.shape[1])
                    )
                    for i in range(len(centers))
                ]
            )
            partitions.append(((labels, centers), lambda D: D))
        return partitions

    def _draw_trajectory(self, universes: np.ndarray):
        traj = np.empty((universes.shape[0], 2))
        for t in range(universes.shape[0]):
            traj[t] = secrets.choice(universes[t])
        return traj.flatten()

    def _make_noisy_counts(self, D: np.ndarray, universes: np.ndarray):
        # calculate for each location of each trajectory the closest location
        # in the genrealized universe for that point in time
        D = D.reshape(D.shape[0], -1, 2)
        D_generalized = np.empty(D.shape)
        for t in range(D.shape[1]):
            D_t = D[:, t, :]
            centers = universes[t]
            distances = np.linalg.norm(D_t[:, np.newaxis] - centers, axis=2)
            closest = np.argmin(
                distances, axis=1
            )  # closest location in generalized dataset
            D_generalized[:, t, :] = centers[closest]
        D_generalized, counts = np.unique(
            D_generalized.reshape(D.shape[0], -1), axis=0, return_counts=True
        )
        # add noise to counts
        lap = Laplace(epsilon=self.eps2, sensitivity=1)
        noisy_counts = np.array([int(np.round(lap.randomise(c))) for c in counts])

        return D_generalized, noisy_counts

    def _location_universes(self, generalized: np.ndarray):
        tuple_shaped = generalized.reshape(generalized.shape[0], -1, 2)
        traj_len = tuple_shaped.shape[1]
        universes = np.empty((traj_len,), dtype=object)
        for t in range(traj_len):
            tuples_t = tuple_shaped[:, t]
            universes[t] = np.unique(tuples_t, axis=0)
        return universes
