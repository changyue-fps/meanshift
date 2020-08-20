#!/usr/bin/env python3

import numpy as np

from argparse import ArgumentParser
from mean_shift.mean_shift import MeanShift


CLUSTERING_EPS = 1e-6


def read_data(filename):
    return np.genfromtxt(filename, delimiter=',')


if __name__ == '__main__':
    parser = ArgumentParser('Script to run mean shift clustering algorithm')
    parser.add_argument('--data', required=True, help='Input data points csv file.')
    parser.add_argument('--band-width', default=1.5, type=float, help='Band width for Mean Shift clustering.')
    
    args = parser.parse_args()

    data = read_data(args.data)

    mean_shift = MeanShift(data, args.band_width, CLUSTERING_EPS)
    clusters = mean_shift.clustering()

    