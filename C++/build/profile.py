import numpy as np
import sys
import math
import numba
import cv2 as cv
from numba import jit
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import scipy.optimize as opt
from skimage.transform import warp, AffineTransform
import time
import cTest as c

def translate(arr, dy, dx):
    """ Translates an array, equivalent to f(x - dx, y - dy). Pads with 0s """
    size = np.int32(arr.shape)

    dx = np.int32(dx)
    dy = np.int32(dy)
    tform = AffineTransform(scale=(1, 1), rotation=0, shear=0, translation=(dx, dy))
    absArr = np.abs(arr)
    angArr = np.angle(arr)
    absOut = warp(absArr, tform.inverse, output_shape=(size[0], size[1]))
    angOut = warp(angArr, tform.inverse, output_shape=(size[0], size[1]))
    out = absOut * np.exp(1j * angOut)
    return out

def timePy(arr, dx, dy):
    start = time.clock()
    translate(arr, dx, dy)
    end = time.clock()
    print(end-start)

def timeCpp(arr, dx, dy):
    start = time.clock()
    out = c.trans(arr, dx, dy)
    end = time.clock()
    print(end - start)
    return out

def padCpp(arr, size):
    start = time.clock()
    out = c.better_pad(arr, size)
    end = time.clock()
    print(end - start)
    return out

def pad(arr, size, c=None):
    """ Pads an array given the size of the output, kwargs: c = [center x, centre y] """
    if c:
        centre = np.int32(c)
    else:
        centre = np.int32([size[0] //2, size[1] // 2])
    arrShape = np.array(arr.shape)
    type = arr.dtype

    len = np.int32(arr.shape)
    posStart = np.int32([centre[0] - len[0] / 2, centre[1] - len[1] / 2])
    posEnd = np.int32([centre[0] + len[0] / 2 - 1, centre[1] + len[1] / 2 - 1])
    if posEnd[0] - posStart[0] <= arr.shape[0]:
        posEnd[0] += 1
    if posEnd[1] - posStart[1] <= arr.shape[1]:
        posEnd[1] += 1

    if arrShape.shape[0] == 2:
        out = np.zeros([size[0], size[1]], dtype=type)
        out[posStart[0]:posEnd[0], posStart[1]:posEnd[1]] = arr
    else:
        out = np.zeros([size[0], size[1], 3], dtype=type)
        out[posStart[0]:posEnd[0], posStart[1]:posEnd[1], :] = arr
    return out

def padPy(arr, size):
    start = time.clock()
    out = pad(arr, size)
    end = time.clock()
    print(end - start)
    return out


if __name__ == "__main__":
    img = np.array(cv.imread("sample1.jpg")[:, :, 0], dtype=np.complex64)
    size = 2000
    out = padCpp(img, size)
    plt.imshow(np.array(np.abs(out), dtype=np.int64))
    plt.show()
    out = padPy(img, [size]*2)
    plt.imshow(np.array(np.abs(out), dtype=np.int64))
    plt.show()

