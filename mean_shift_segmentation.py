#!/usr/bin/env python3

import cv2 as cv
import numpy as np

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser('Script to run mean shift segmentation using OpenCV')
    parser.add_argument('--image', required=True, help='Image to segment')
    parser.add_argument('--output', required=True, help='Output image')
    parser.add_argument('--show', action='store_true', help='Flag to display image')

    args = parser.parse_args()

    image = cv.imread(args.image)
    if image is None:
        print(f'Failed to read image: {args.image}')
        exit(-1)

    # Mean shift filtering.
    res = cv.pyrMeanShiftFiltering(src=image, sp=32, sr=32, maxLevel=2)

    cv.imwrite(args.output, res)

    if args.show:
        cv.namedWindow('source', cv.WINDOW_NORMAL)
        cv.imshow('source', image)
        cv.namedWindow('segmentation', cv.WINDOW_NORMAL)
        cv.imshow('segmentation', res)
        cv.waitKey(0)
        cv.destroyAllWindows()
