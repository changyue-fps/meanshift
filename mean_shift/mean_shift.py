import numpy as np
from cluster import Cluster
from clusters_builder import ClustersBuilder
from utils import compute_euclidean_distance, compute_gaussian


class MeanShift:
    def __init__(self, points, sigma, cluster_eps):
        self._points = points
        self._sigma = sigma
        self._cluster_eps = cluster_eps

    def cluster(self):
        """
        Cluster points based on Mean Shift algorithm.
        """

        builder = ClustersBuilder(self._points, self._cluster_eps)

        while not builder.finished():
            for i in range(0, self._points.shape[0]):
                if builder.finished_point(i):
                    continue

                point_to_shift = builder.shifted_point(i)

                # Vectorized computation to speed up clustering.
                dists = np.sqrt(np.sum(np.square(self._points - point_to_shift), axis=1))
                gaussian_weights = compute_gaussian(dists, self._sigma)
                gaussian_weights = gaussian_weights * (dists <= 3 * self._sigma)
                weights = np.tile(gaussian_weights, (point_to_shift.shape[0], 1))
                new_point = np.multiply(weights.transpose(), self._points).sum(axis=0) / np.sum(gaussian_weights)
                
                #new_point = np.zeros(point_to_shift.shape)
                #weights = 0.0
                #for point in self._points:
                #    dist = compute_euclidean_distance(point_to_shift, point)
                #
                #    # Points within 3*sigma distance contribute to weight computation.
                #    if dist <= 3 * self._sigma:
                #        weight = compute_gaussian(dist, self._sigma)
                #        new_point += point * weight
                #        weights += weight
                #
                #new_point /= weights

                builder.shift_point(i, new_point)

        return builder.build_clusters()
