from mean_shift.cluster import Cluster
from mean_shift.util import compute_euclidean_distance

class ClustersBuilder:
    def __init__(self, points, eps):
        self._original_points = points
        self._shifted_points = points
        self._shifting = np.full(points.shape[0], True, dtype=bool)
        self._cluster_eps = eps
        self._shifting_eps = eps / 10.0
    
    def get_shifted_point(self, index):
        return self._shifted_points[index]
    
    def finished_point(self, index):
        return not self._shifting[idex]

    def finished(self):
        return np.count_nonzero(self._shifting == True) == 0

    def shift_point(self, index, new_point):
        euclidean_dist = compute_euclidean_distance(self._shifted_points[index], new_point)
        if euclidean_dist <= self._shifting_eps:
            self._shifting[index] = False
        else:
            self._shifted_points[index] = new_point

    def build_clusters(self):
        clusters = []
        for i in range(0, self._shifted_points.shape[0]):
            shifted_point = self._shifted_points[i]

            # Try to add a point to an existing cluster.
            added = False
            for cluster in clusters:
                centroid = clusters[cluster_idx].centroid()
                dist_to_centroid = compute_euclidean_distance(centroid, shifted_point)
                if dist_to_centroid <= self._cluster_eps:
                    cluster.add_point(self._original_points[i])
                    added = True
                    break

            # Try to create a new cluster if the point does not belong to any existing ones.
            if not added:
                cluster = Cluster(shifted_point)
                cluster.add_point(self._original_points[i])
                clusters.append(cluster)
        
        return clusters
