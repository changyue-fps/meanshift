import numpy as np

def compute_euclidean_distance(point1, point2):
    """
    Compute Euclidean distance(s) between points.

    Args:
        point1: Left-hand side `numpy.ndarray`.
        point2: Right-hand side `numpy.ndarray`.

    Returns:
        The computed Euclidean distance(s).
    """

    return np.sqrt(np.sum(np.square(point1 - point2), axis=0, keepdims=True))


def compute_gaussian(distance, sigma):
    """
    Compute Gaussian with given `distance` and `sigma`.

    Args:
        distance: Euclidean distance.
        sigma: Gaussian standard deviation.

    Returns:
        The computed Gaussian value.
    """

    return np.exp(-0.5 * np.square(distance / sigma)) / (sigma * np.sqrt(2 * np.pi))