import numpy as np

class Cluster:
    """
    Class represents a set of points that belong to the same cluster.
    Cluster contains data points and centroid.
    """

    def __init__(self, centroid):
        self._centroid = centroid
        self._points = []


    def add_point(self, point):
        """
        Add `point` to the cluster.
        """

        self._points.append(point)


    @property
    def centroid(self):
        return self._centroid


    @property
    def points(self):
        return self._points
