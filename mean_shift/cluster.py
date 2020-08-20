import numpy as np

class Cluster:
    def __init__(self, centroid):
        assert isinstance(centroid, np.ndarray), 'Centroid should be type `numpy.ndarray`'
        assert centroid.ndim == 1, 'Cenroid should be a one-dimensional array'

        self._centroid = centroid
        self._points = np.empty((0, centroid.shape[0]))
    
    @property
    def centroid(self):
        return self._centroid

    def add_point(self, point):
        self.points = np.vstack((self.points, point))
    
    def compute_sse(self):
        mat = self.points - self.centroid
        return np.sqrt(np.sum(np.square(mat)))
        