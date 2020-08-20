import numpy as np

def compute_euclidean_distance(point1, point2):
    assert point1.shape == point2.shape, 'Arguments shape must be the same!'

    return np.sqrt(np.sum(np.square(point1 - point2)))