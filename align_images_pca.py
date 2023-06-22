# Implementation of Eigenhits
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2022

import math

import numpy as np
# to compute angles
from math import atan2, cos, sin, sqrt, pi
import cv2

from scipy.ndimage import gaussian_filter


def align_image(input_image, mask=False, threshold=25):
    src_copy = np.copy(input_image)
    if len(src_copy.shape) > 2:
        gray = cv2.cvtColor(src_copy, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy(src_copy)
    gray = gaussian_filter(gray, sigma=1)
    # convert img into binary
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    nz = np.nonzero(bw)
    my_len = len(nz[0])
    data_pts = np.empty((my_len, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = nz[1][i]
        data_pts[i, 1] = nz[0][i]
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    angle = 180 * angle / math.pi

    (cX, cY) = cntr
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    (h, w) = src_copy.shape[:2]

    if mask:
        src_copy[bw < 1] = 0

    rotated = cv2.warpAffine(src_copy, M, (w, h))

    xx = w / 2 - cX
    yy = h / 2 - cY
    M = np.float32([[1, 0, xx], [0, 1, yy]])

    shifted = cv2.warpAffine(rotated, M, (rotated.shape[1], rotated.shape[0]))
    return shifted
