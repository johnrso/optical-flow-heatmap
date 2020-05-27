#!/usr/bin/env python

# template code taken from opt_flow.py under https://github.com/opencv/opencv/tree/master/samples/python

import numpy as np
import cv2

temp = []
total = []

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_heatmap(flow):
    global temp, total

    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)

    fx, fy = flow[:,:,0], flow[:,:,1]
    v = np.sqrt(fx*fx+fy*fy)

    hsv[..., 0] = get_norm(v)
    hsv[..., 1] = 255
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_heatmap_sum(flow):
    global total

    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)

    fx, fy = flow[:,:,0], flow[:,:,1]
    v = np.sqrt(fx*fx+fy*fy)

    total = np.add(total, v)
    norm = cv2.normalize(total,None,0,255,cv2.NORM_MINMAX)
    norm = np.add(np.full((h, w), 255), -norm)

    hsv[...,0] = norm
    hsv[...,1] = 255
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def get_norm(v):
    global temp, total
    assert v.shape == temp.shape[:2]

    v = np.maximum(0, v - np.sum(v) /(v.shape[0] * v.shape[1]))
    temp = np.dstack((temp, v))

    if temp.shape[2] > 30:
        temp = np.delete(temp, 0, 2)
    total = np.add(total, v)

    sum = np.sum(temp, axis = 2)
    norm = cv2.normalize(sum,None,0,255,cv2.NORM_MINMAX)

    return np.full(temp.shape[:2], 255) - norm

def main():
    global temp, total

    cam = cv2.VideoCapture(0)
    _ret, prev = cam.read()

    temp = np.zeros((prev.shape[0], prev.shape[1]), np.uint8)
    total = np.zeros((prev.shape[0], prev.shape[1]), np.uint8)

    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        _ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        heatmap = draw_heatmap(flow)
        overlay = cv2.addWeighted(img,0.7,heatmap,0.3,0)

        prevgray = gray

        cv2.imshow('heatmap', heatmap)
        cv2.imshow('overlaid', overlay)

        ch = cv2.waitKey(1)
        if ch == 27:
            break
        elif ch == 32:
            temp = np.zeros((prev.shape[0], prev.shape[1]), np.uint8)

    print('exiting...')

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
