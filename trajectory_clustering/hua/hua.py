import secrets
import numpy as np
from numpy import float64, floating, int64
from sklearn.cluster import KMeans
from diffprivlib.mechanisms import Exponential, Laplace


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
        generalized = self._dp_location_generalization(D)
        release = self._dp_release(D, generalized)

        return release

    def _dp_location_generalization(self, D: np.ndarray):
        kmeans = KMeans(n_clusters=self.m).fit(D)
        p_opt = (kmeans.labels_, kmeans.cluster_centers_)
        modifications = self._phi_sub_optimal(D, p_opt)
        s_partitions = self._s_kmeans_partitions(D)

        p_opt_modifications = self.apply_modifications(D, p_opt, modifications)

        partitions = p_opt_modifications + [(p_opt, lambda D: D)] + s_partitions

        p_opt_mean_dist = self._mean_distance(D, p_opt)
        utility_score = lambda x: p_opt_mean_dist / self._mean_distance(
            x[1](D), x[0]
        )  # x.mean_distance() may be zero, but highly unlikely for real-world data

        utilities = [utility_score(p) for p in partitions]
        exp = Exponential(
            epsilon=self.eps1, sensitivity=1, utility=utilities, candidates=partitions
        )
        ((_, centers), _) = exp.randomise()  # type: ignore

        return centers

    def _dp_release(self, D: np.ndarray, generalized: np.ndarray):

        # filter out non-positive counted trajectories
        noisy_counts = list(
            filter(lambda c: c > 0, self._make_noisy_counts(D, generalized))
        )

        universes = self._location_universes(generalized)
        size_omega = np.prod([u.shape[0] for u in universes])
        size_remaining_omega = size_omega - generalized.shape[0]
        total_count = 0
        release_trajects = np.empty((0, generalized.shape[1]))
        release_counts = []
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

            if num_i > 0:
                rand_trajects = np.array(
                    [self._draw_trajectory(universes) for _ in range(num_i)]
                )
                rand_counts = secrets.randbelow(ci - cj) + cj

                release_trajects = np.concatenate((release_trajects, rand_trajects))
                release_counts.append(rand_counts)
                total_count += num_i + rand_counts

            release_trajects = np.concatenate((release_trajects, [generalized[i]]))
            release_counts.append(noisy_counts[i])
            total_count += noisy_counts[i]

            if total_count >= D.shape[0]:
                break

        return release_trajects, release_counts

    def _phi_sub_optimal(self, D: np.ndarray, p_opt) -> list[list[Modification]]:
        indiv_mods = self._phi_sub_optimal_individual(D, p_opt)
        result = [[indiv_mods[0]]]

        for indiv_mod in indiv_mods[1:]:
            tmp = [[indiv_mod]]

            for mod in result:
                if not any(indiv_mod.id == m.id for m in mod):
                    new_mod = mod.copy() + [indiv_mod]
                    tmp.append(new_mod)

            result += tmp
            result.sort(key=lambda x: sum(float(m.distance) for m in x))
            if len(result) > self.phi:
                result = result[0 : self.phi]

        return result

    def _phi_sub_optimal_individual(self, D: np.ndarray, p_opt) -> list[Modification]:
        labels, centers = p_opt
        modifications: list[Modification] = []
        for i, traj in enumerate(D):
            label = labels[i]
            dis_opt = np.linalg.norm(traj - centers[label])

            # move to other clusters and calculate distance change
            for k in set(labels) - {label}:
                dis = np.linalg.norm(traj - centers[k])
                modifications.append(Modification(i, label, k, dis - dis_opt))

        return sorted(modifications, key=lambda x: float(x.distance))[0 : self.phi]

    def _s_kmeans_partitions(self, D: np.ndarray):
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

    def _make_noisy_counts(self, D: np.ndarray, generalized: np.ndarray):
        # calculate for each tr in D the distance to each center
        centers = generalized
        distances = np.linalg.norm(D[:, np.newaxis] - centers, axis=2)
        # for each center, get the closest trajectory
        closest = np.argmin(distances, axis=1)
        # for each center, count how many trajectories are closest to it
        counts = np.bincount(closest, minlength=centers.shape[0])
        # add Laplace noise
        lap = Laplace(epsilon=self.eps2, sensitivity=1)
        noisy_counts = [lap.randomise(c) for c in counts]

        # noisy counts should be discrete, the paper does not mention this
        # we round to the nearest integer here
        noisy_counts_discrete = [int(np.round(c)) for c in noisy_counts]
        return sorted(noisy_counts_discrete, reverse=True)

    def _location_universes(self, generalized: np.ndarray):
        tuple_shaped = generalized.reshape(generalized.shape[0], -1, 2)
        traj_len = tuple_shaped.shape[1]
        universes = np.empty((traj_len,), dtype=object)
        for t in range(traj_len):
            tuples_t = tuple_shaped[:, t]
            universes[t] = np.unique(tuples_t, axis=0)
        return universes
