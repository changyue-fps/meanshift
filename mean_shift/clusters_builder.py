import numpy as np

from mean_shift.cluster import Cluster
from mean_shift.utils import compute_euclidean_distance

class ClustersBuilder:
    """
    Class that builds shifted points into clusters.
    """

    def __init__(self, points, cluster_eps):
        self._original_points = points
        self._shifted_points = np.copy(points)
        self._shifting = np.full(points.shape[1], True, dtype=bool)
        self._cluster_eps = cluster_eps
        self._shifting_eps = cluster_eps / 10.0


    def get_point(self, index):
        """
        Returns shifted point at the given index.
        """

        return self._shifted_points[:,index:index+1]


    def shift_point(self, index, new_point):
        """
        Shift point at the given `index` to new position `new_point`.

        Args:
            index: Index of point to be shifted.
            new_point: New position.
        """

        dist = compute_euclidean_distance(self._shifted_points[:,index:index+1], new_point)
        if dist <= self._shifting_eps:
            self._shifting[index] = False

        self._shifted_points[:,index:index+1] = new_point


    def converged_at(self, index):
        """
        Returns True if point at the given `index` has converged, False otherwise.
        """

        return not self._shifting[index]


    def converged(self):
        """
        Returns True if all points have converged, False otherwise.
        """
        return np.count_nonzero(self._shifting == True) == 0


    def build_clusters(self):
        """
        Build clusters once points have been shifted.

        Returns:
            A list of clusters.
        """

        clusters = []
        for i in range(0, self._shifted_points.shape[1]):
            shifted_point = self.get_point(i)

            # Try to add a point to an existing cluster.
            added = False
            for cluster in clusters:
                dist = compute_euclidean_distance(cluster.centroid, shifted_point)
                if dist <= self._cluster_eps:
                    cluster.add_point(self._original_points[:,i:i+1])
                    added = True
                    break

            # Try to create a new cluster if the point does not belong to any existing ones.
            if not added:
                cluster = Cluster(shifted_point)
                cluster.add_point(self._original_points[:,i:i+1])
                clusters.append(cluster)
        
        return clusters
