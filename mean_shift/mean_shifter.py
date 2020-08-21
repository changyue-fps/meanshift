#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import time

from argparse import ArgumentParser
from mean_shift import MeanShift


CLUSTERING_EPS = 1e-6


def read_data(filename):
    return np.genfromtxt(filename, delimiter=',')


def plot_clusters(clusters):
    points = []
    cluster_ids = []
    for i in range(0, len(clusters)):
        points.extend(clusters[i].points)
        cluster_ids.extend([i] * len(clusters[i].points))
    points = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(points[:,0], points[:,1], c=cluster_ids, s=50)

    for cluster in clusters:
        centroid = cluster.centroid
        ax.scatter(centroid[0], centroid[1], c='red', s=50, marker='+')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter)
    fig.savefig('mean_shift_clusters')


if __name__ == '__main__':
    parser = ArgumentParser('Script to run mean shift clustering algorithm')
    parser.add_argument('--data', required=True, help='Input data points csv file.')
    parser.add_argument('--sigma', default=1.5, type=float, help='Band width for Mean Shift clustering.')

    args = parser.parse_args()

    data = read_data(args.data)

    start_time = time.time()
    mean_shift = MeanShift(data, args.sigma, CLUSTERING_EPS)
    clusters = mean_shift.cluster()
    end_time = time.time()
    print(f'MeanShift run time {end_time - start_time} seconds.')

    plot_clusters(clusters)
