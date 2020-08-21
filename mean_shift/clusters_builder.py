import numpy as np

from cluster import Cluster
from utils import compute_euclidean_distance

class ClustersBuilder:
    """
    Class that builds shifted points into clusters.
    """

    def __init__(self, points, eps):
        self._original_points = points
        self._shifted_points = np.copy(points)
        self._shifting = np.full(points.shape[0], True, dtype=bool)
        self._cluster_eps = eps
        self._shifting_eps = eps / 10.0
    

    def shifted_point(self, index):
        return self._shifted_points[index]
    

    def finished_point(self, index):
        """
        Returns True if point at the given `index` has finished shifting, False otherwise.
        """

        return not self._shifting[index]


    def finished(self):
        """
        Returns True if all points have finished shifting, False otherwise.
        """
        return np.count_nonzero(self._shifting == True) == 0


    def shift_point(self, index, new_point):
        """
        Shift point at the given `index` to new position `new_point`.

        Args:
            index: Index of point to be shifted.
            new_point: New position.
        """

        dist = compute_euclidean_distance(self._shifted_points[index], new_point)
        if dist <= self._shifting_eps:
            self._shifting[index] = False
        #else:
        #    self._shifted_points[index] = new_point
        self._shifted_points[index] = new_point


    def build_clusters(self):
        """
        Build clusters once points have been shifted.

        Returns:
            A list of clusters.
        """

        clusters = []
        for i in range(0, self._shifted_points.shape[0]):
            shifted_point = self._shifted_points[i]

            # Try to add a point to an existing cluster.
            added = False
            for cluster in clusters:
                dist = compute_euclidean_distance(cluster.centroid, shifted_point)
                if dist <= self._cluster_eps:
                    cluster.add_point(self._original_points[i])
                    added = True
                    break

            # Try to create a new cluster if the point does not belong to any existing ones.
            if not added:
                cluster = Cluster(shifted_point)
                cluster.add_point(self._original_points[i])
                clusters.append(cluster)
        
        return clusters
