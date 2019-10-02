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


def circle(size, radius):
    """ Draws full circle using GPU, kwargs: dtype, value = value in circle """

    if 2*radius > size[0] or 2*radius > size[1]:
        raise Exception('Radius too large: {}'.format(radius))

    arr = np.zeros([size[0], size[1], 3])
    arr = cv.circle(arr, (int(size[0]/2), int(size[1]/2)), int(radius), (1, 1, 1), -1)
    return arr


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


def crop(arr, size, c=None):
    """ Returns cropped image given the desired size, kwargs: c = [centre x, centre y] """
    inType = arr.dtype
    inSize = np.array(arr.shape)

    centre = np.int32([inSize[0]/2, inSize[1]/2])

    len = np.int32(size)
    posStart = np.int32([centre[0] - len[0] // 2, centre[1] - len[1] // 2])
    posEnd = [posStart[0]+len[0], posStart[1]+len[1]]
    if inSize.shape[0] == 2:
        out = np.array(arr[posStart[0]:posEnd[0], posStart[1]:posEnd[1]])
    else:
        out = np.array(arr[posStart[0]:posEnd[0], posStart[1]:posEnd[1], :])
    return out


def translate(arr, dy, dx):
    """ Translates an array, equivalent to f(x - dx, y - dy). Pads with 0s """
    size = np.int32(arr.shape)

    dx = np.int32(dx)
    dy = np.int32(dy)
    tform = AffineTransform(scale=(1, 1), rotation=0, shear=0, translation=(dy, dx))
    absArr = np.abs(arr)
    angArr = np.angle(arr)
    absOut = warp(absArr, tform.inverse, output_shape=(size[0], size[1]))
    angOut = warp(angArr, tform.inverse, output_shape=(size[0], size[1]))
    out = absOut * np.exp(1j * angOut)
    return out


def E(arr, xc, yc, R):
    """ Evaluates the sum of points around the circumference of a given circle """
    circ = np.int32(np.ceil(2 * np.pi * R))
    angles = np.linspace(0, 2*np.pi, circ)
    sum = np.float64(0)
    for i in range(angles.shape[0]):
        x = int(R * np.cos(angles[i]))
        y = int(R * np.sin(angles[i]))
        sum = sum + arr[x + xc, y + yc]
    return sum


def checkR(start, end, centrePos, FIGrad):
    """ Iterates through a sample of R values evaluating E1 for each one, returns radii and E1 values """
    Rs = np.empty(end-start, dtype=np.int32)
    Es = np.empty(end-start, dtype=np.float64)
    for i, R in enumerate(range(start, end)):
        Rs[i] = R
        Es[i] = E(FIGrad, centrePos[0], centrePos[1], R)
    return Rs, Es


def gridE(arr, gridSize, centre, R):
    """ Determines a matrix of values for E about a given set of coordinates """
    c = gridSize // 2
    out = np.zeros([gridSize, gridSize], np.float64)
    for i in range(-c, c):
        for i1 in range(-c, c):
            EVal = E(arr, centre[0]+i, centre[1]+i1, R)
            out[c+i, c+i1] = EVal
    return out


def lineE(arr, R, centre):
    """ Performs one iteration of an axis by axis line search """
    R = int(R)
    stripX = np.zeros(2 * R)
    stripY = np.zeros(2 * R)
    for i, r in enumerate(range(-R, R)):
        stripX[i] = E(arr, int(centre[0] + r), int(centre[1]), R)
    xPos = np.argmax(stripX)
    centre = [int(xPos - R + centre[0]), int(centre[1])]
    for i, r in enumerate(range(-R, R)):
        stripY[i] = E(arr, int(centre[0]), int(centre[1] + r), R)
    yPos = np.argmax(stripY)
    centre = [int(centre[0]), int(yPos - R + centre[1])]
    return centre


def getStd(arr, R, centre):
    """ Returns the standard deviation in E by computing E for all points R from the centre """
    distrib = np.array([])
    for i, r in enumerate(range(-R, R + 1)):
        EVal = E(arr, centre[0] + r, centre[1], R)
        temp = []
        for i1 in range(int(np.ceil(EVal))):
            temp.append([r])
        distrib = np.append(distrib, temp)
    std = np.std(distrib)
    return std


def getHigh(centre, std):
    """ Returns all points within 1 std of the max E """
    highCoords = np.array([[centre[0], centre[1]]])
    for i, r0 in enumerate(range(1, int(std*0.2))):
        quantisation = np.ceil(2 * np.pi * r0)
        angles = np.linspace(0, 2 * np.pi, quantisation)
        for i1, angle in enumerate(angles):
            xc = int(r0 * np.cos(angle)) + centre[0]
            yc = int(r0 * np.sin(angle)) + centre[1]
            highCoords = np.append(highCoords, [[xc, yc]], axis=0)
    return highCoords


def whereArr(arrLarge, arrSub):
    """ Returns the index of the location of the sub array in the main array """
    indices = []
    for i in range(arrLarge.shape[0]):
        if np.array_equal(arrLarge[i, :], arrSub):
            indices.append(i)
    indices = np.array(indices)
    return indices


def isInArr(arrLarge, arrSub):
    """ Returns whether a sub array is in the larger array """
    for i in range(arrLarge.shape[0]):
        if np.array_equal(arrLarge[i, :], arrSub):
            return True
    return False


def progbar(curr, total, name):
    """ Displayes the progress on the CMD """
    frac = curr/total
    sys.stdout.write('\r' + '{}: {:>7.2%}'.format(name, frac))


def progbar_time(curr, total, name, start):
    """ Displayes the progress on the CMD """
    frac = curr/total
    if frac != 0:
        timeTaken = time.time() - start
        m = frac / timeTaken
        timeRemaining = np.round(((1 / m) - timeTaken) / 60)
    else:
        timeRemaining = '-'
    sys.stdout.write('\r                                                                                ')
    sys.stdout.write('\r' + '{}: {:>7.2%} Time Remaining: {} mins'.format(name, frac, timeRemaining))


def transMatrix(arr, params):
    """ Performs an affine transform on the input coordinates """
    angle, transX, transY = params
    c, s = np.cos(angle), np.sin(angle)
    rotate = np.array(((c, -s), (s, c)))
    outArr = np.matmul(rotate, arr) + [transX, transY]
    return outArr


def SSD(params, realF, realLED):
    """ Returns the square distance between the real and transformed coordinates """
    out = 0
    for i in range(len(realF)):
        model = transMatrix(realLED[i], params)
        out += np.sqrt(np.sum(np.square(realF[i] - model)))
    return out


def RANSACTransform(inDat, outDat, threashProportion=0.5, minSamples=3, radiusAccepted=20, maxTrials=100):
    """ Performs RANSAC optimisation with SSD between model and data """
    inliers = [True] * len(inDat)
    params = np.zeros(3)
    threash = threashProportion * len(inDat)
    for i in range(maxTrials):
        selectIndices = np.random.randint(0, len(inDat), minSamples)
        inDatSelect = np.take(inDat, selectIndices, axis=0)
        outDatSelect = np.take(outDat, selectIndices, axis=0)
        dict_Fit = opt.minimize(SSD,
                                 params,
                                 method='nelder-mead',
                                 args=(outDatSelect, inDatSelect))

        k = 0
        if dict_Fit.success:
            params = dict_Fit.x
            for i1 in range(len(inDat)):
                fit = SSD(params, [inDat[i1]], [outDat[i1]])
                if fit <= radiusAccepted ** 2:
                    inliers[i1] = True
                    k += 1
                else:
                    inliers[i1] = False

        if k >= threash:
            break
    return params, inliers


def genBayer(size, colour):
    """ Generates a bayer filter array to be used with RGB sensors """
    subArr = np.zeros([2, 2])
    if colour == 0:
        subArr[0, 0] = 1
    elif colour == 1:
        subArr[0, 1] = 1
        subArr[1, 0] = 1
    else:
        subArr[1, 1] = 1

    invArr = np.ones((size[0], size[1]))
    numTiles = [int(size[0] / 2), int(size[1] / 2)]
    fullArr = np.tile(subArr, (numTiles[0], numTiles[1]))
    fullArr = pad(fullArr, [size[0], size[1]]) # Not good
    invArr = invArr - fullArr
    return fullArr, invArr


def fft2Phase(arr):
    fft = np.fft.fft2(arr)
    fftShift = np.fft.fftshift(fft)
    amp = np.abs(fftShift)
    phase = np.angle(fftShift)
    phase = np.mod(phase, np.pi)
    total = amp * np.exp(1j * phase)
    return total


def ifft2Phase(arr):
    fft = np.fft.ifft2(arr)
    fftShift = np.fft.fftshift(fft)
    amp = np.abs(fftShift)
    phase = np.angle(fftShift)
    phase = np.mod(phase, 2*np.pi)
    total = amp * np.exp(1j * phase)
    return total