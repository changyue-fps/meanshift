import numpy as np

def compute_euclidean_distance(point1, point2):
    """
    Compute Euclidean distance between two points.

    Args:
        point1: Left-hand side `numpy.ndarray` point.
        point2: Right-hand side `numpy.ndarray` point.

    Returns:
        The computed Euclidean distance.
    """

    assert point1.shape == point2.shape, 'Arguments must have the same shape!'

    return np.sqrt(np.sum(np.square(point1 - point2)))


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