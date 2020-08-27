#!/usr/bin/env python3

import cv2 as cv
import numpy as np

from argparse import ArgumentParser


RECTANGLE_COLOR = (0, 255, 0)

# Global variales for mouse event.
draw = False
upper_left = (-1, -1)
lower_right = (-1, -1)
image_drew = None


def on_mouse_event(event, x, y, flags, param):
    global draw, upper_left, lower_right, image_drew

    if event == cv.EVENT_LBUTTONDOWN:
        draw = True
        upper_left = (x, y)
    elif event == cv.EVENT_MOUSEMOVE:
        if draw:
            image_drew = param.copy()
            cv.rectangle(image_drew, upper_left, (x, y), RECTANGLE_COLOR, 2)
    elif event == cv.EVENT_LBUTTONUP:
        draw = False
        lower_right = (x, y)
        image_drew = param.copy()
        cv.rectangle(image_drew, upper_left, (x, y), RECTANGLE_COLOR, 2)


def initialize_window(initial_frame):
    global image_drew

    window_name = 'Initialization - Place Rectangle'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window_name,on_mouse_event, initial_frame)

    image_drew = initial_frame.copy()
    while True:
        cv.imshow(window_name, image_drew)

        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()

def track_object(filename):
    vc = cv.VideoCapture(filename)

    # Read the first frame
    retval, frame = vc.read()
    if not retval:
        print(f'Failed to read video frames from file {filename}')
        return
    
    initialize_window(frame)

    col = upper_left[0]
    row = upper_left[1]
    width = lower_right[0] - upper_left[0]
    height = lower_right[1] - upper_left[1]
    track_window = (col, row, width, height)

    # Track RoI
    roi = frame[row:row+height, col:col+width]
    roi_in_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(roi_in_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([roi_in_hsv], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    cv.namedWindow('tracked', cv.WINDOW_NORMAL)
    while True:
        retval, frame = vc.read()
        if not retval:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        retval, track_window = cv.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window
        image = cv.rectangle(frame, (x, y), (x+w, y+h), RECTANGLE_COLOR, 2)
        cv.imshow('tracked', image)

        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
    
    cv.destroyAllWindows()
    vc.release()

if __name__ == '__main__':
    parser = ArgumentParser('Script to run Mean Shift object tracking based on an initialized window.')
    parser.add_argument('--video', required=True, help='Video file.')

    args = parser.parse_args()

    track_object(args.video)
    