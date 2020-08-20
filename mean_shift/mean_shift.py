import numpy as np
from mean_shift.cluster import Cluster
from mean_shift.clusters_builder import ClustersBuilder
from mean_shift.util import compute_euclidean_distance


class MeanShift:
    def __init__(self, points, band_width, cluster_eps):
        self._points = points
        self._band_width = band_width
        self._builder = ClustersBuilder(points, cluster_eps)

    def clustering(self):
        while not self._builder.finished():
            for i in range(0, self._points.shape[0]):
                if self._builder.finished_point(i):
                    continue

                point_to_shift = self._builder.get_shifted_point(i)
                new_point = np.zeros(point_to_shift.shape)
                weights = 0.0
                for point in points:
                    euclidean_dist = compute_euclidean_distance(point_to_shift, point)
                    if euclidean_dist <= 3 * self._band_width:
                        weight = np.exp(-0.5 * np.square(euclidean_dist / self._band_width)) / (self._band_width * np.sqrt(2 * np.pi))
                        new_point += point * weight
                        weights += weight
                
                new_point /= weights
                self._builder.shift_point(i, new_point)

        return self._builder.build_clusters()
