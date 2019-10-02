import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import sys
import math
import numba
import numba.cuda as cuda
import cv2 as cv
from numba import jit
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import scipy.optimize as opt


def setType(value, dtype):
    """ Returns pointer to scalar of given dtype """
    if dtype == 'complex64':
        return np.complex64(value)
    elif dtype == 'float32':
        return np.float32(value)
    elif dtype == 'int16':
        return np.int16(value)
    elif dtype == 'int32':
        return np.int32(value)
    else:
        types = 'complex64, float32, int16, int32'
        raise Exception('Type {} not supported \n Supported types: \n {}'.format(dtype, types))

def getThreads(xSize, ySize):
    """ Returns threads per block """
    if xSize <= 32:
        x = xSize
    else:
        x = 32
    if ySize <= 32:
        y = ySize
    else:
        y = 32
    return x, y


@cuda.jit
def multGPU(arr1, arr2, out, size):
    x, y = cuda.grid(2)
    if x < size[0] and y < size[1]:
        out[x, y] = arr1[x, y] * arr2[x, y]

def mult(arr1, arr2):
    """ Multiplies two matrices together element wise """
    if arr1.dtype != arr2.dtype:
        return Exception('Types {} and {} cannot be cast together'.format(arr1.dtype, arr2.dtype))

    type = arr1.dtype
    size = np.int32(arr1.shape)
    out = np.empty([size[0], size[1]], dtype=type)#cuda.device_array((size[0], size[1]), dtype=type)

    threadX, threadY = getThreads(size[0], size[1])
    threadsperblock = (threadX, threadY)
    blockspergrid_x = math.ceil(out.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(out.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    multGPU[blockspergrid, threadsperblock](arr1, arr2, out, size)
    return out#.copy_to_host()


@cuda.jit
def circleGPU(arr, size, apRad, v):
    x, y = cuda.grid(2)
    if x < size[0] and y < size[1]:
        if (((x - apRad) ** 2 + (y - apRad) ** 2)**0.5) < apRad:
            arr[x - apRad + int(size[0] // 2), y - apRad + int(size[1] // 2)] = v

def circle(size, radius, dtype='float32', value=1):
    """ Draws full circle using GPU, kwargs: dtype, value = value in circle """

    if 2*radius > size[0] or 2*radius > size[1]:
        raise Exception('Radius too large: {}'.format(radius))

    v = setType(value, dtype)
    apRad = np.int32(radius)
    arr = np.empty([size[0], size[1]], dtype=dtype)#cuda.device_array((size[0], size[1]), dtype=dtype)
    size = np.int32(size)

    threadX, threadY = getThreads(size[0], size[1])
    threadsperblock = (threadX, threadY)
    blockspergrid_x = math.ceil(arr.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(arr.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    circleGPU[blockspergrid, threadsperblock](arr, size, apRad, v)
    return arr#.copy_to_host()


@cuda.jit
def padGPU(arr, out, posStart, posEnd):
    x, y = cuda.grid(2)
    if x < out.shape[0] and y < out.shape[1]:
        if x >= posStart[0] and x <= posEnd[0]:
            if y >= posStart[1] and y <= posEnd[1]:
                out[x, y] = arr[x - posStart[0], y - posStart[1]]

def pad(arr, size, c=None):
    """ Pads an array given the size of the output, kwargs: c = [center x, centre y] """

    inType = arr.dtype
    out = np.empty([size[0], size[1]], dtype=inType)#cuda.device_array((size[0], size[1]), dtype=inType)

    if c:
        centre = np.int32(c)
    else:
        centre = np.int32([size[0] //2, size[1] // 2])

    len = np.int32(arr.shape)
    posStart = np.int32([centre[0] - len[0] // 2, centre[1] - len[1] // 2])
    posEnd = np.int32([centre[0] + len[0] // 2 - 1, centre[1] + len[1] // 2 - 1])

    if posStart[0] < 0 or posStart[1] < 0:
        raise Exception('Invalid start point: {}'.format(posStart))
    if posEnd[0] > out.shape[0] or posEnd[1] > out.shape[1]:
        raise Exception('Invalid end point: {}'.format(posEnd))

    threadX, threadY = getThreads(size[0], size[1])
    threadsperblock = (threadX, threadY)
    blockspergrid_x = math.ceil(out.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(out.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    padGPU[blockspergrid, threadsperblock](arr, out, posStart, posEnd)
    return out#.copy_to_host()


@cuda.jit
def cropGPU(arr, out, posStart):
    x, y = cuda.grid(2)
    if x < out.shape[0] and y < out.shape[1]:
        out[x, y] = arr[x + posStart[0], y + posStart[1]]

def crop(arr, size, c=None):
    """ Returns cropped image given the desired size, kwargs: c = [centre x, centre y] """
    inType = arr.dtype
    inSize = arr.shape
    out = np.empty([size[0], size[1]], dtype=inType)#cuda.device_array((size[0], size[1]), dtype=inType)

    if c:
        centre = np.int32(c)
    else:
        centre = np.int32([inSize[0]/2, inSize[1]/2])

    len = np.int32(size)
    posStart = np.int32([centre[0] - len[0] // 2, centre[1] - len[1] // 2])

    if posStart[0] < 0 or posStart[1] < 0:
        raise Exception('Invalid start point: {}'.format(posStart))

    threadX, threadY = getThreads(size[0], size[1])
    threadsperblock = (threadX, threadY)
    blockspergrid_x = math.ceil(out.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(out.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cropGPU[blockspergrid, threadsperblock](arr, out, posStart)
    return out#.copy_to_host()


@cuda.jit
def translateGPU(arr, out, size, dx, dy):
    x, y = cuda.grid(2)
    if x < out.shape[0] and y < out.shape[1]:
        if x - dx < 0 or x - dx >= size[0] or y - dy < 0 or y - dy >= size[1]:
            out[x, y] = 0
        else:
            out[x, y] = arr[x - dx, y - dy]

def translate(arr, dy, dx):
    """ Translates an array, equivalent to f(x - dx, y - dy). Pads with 0s """
    inType = arr.dtype
    size = np.int32(arr.shape)
    out = np.empty([size[0], size[1]], dtype=inType)#cuda.device_array((size[0], size[1]), dtype=inType)

    dx = np.int32(dx)
    dy = np.int32(dy)

    arrC = np.ascontiguousarray(arr)

    threadX, threadY = getThreads(size[0], size[1])
    threadsperblock = (threadX, threadY)
    blockspergrid_x = math.ceil(out.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(out.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    translateGPU[blockspergrid, threadsperblock](arrC, out, size, dx, dy)
    return out#.copy_to_host()


#@jit
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


#@jit#(nopython=True, parallel=True)
def checkR(start, end, centrePos, FIGrad):
    """ Iterates through a sample of R values evaluating E1 for each one, returns radii and E1 values """
    Rs = np.empty(end-start, dtype=np.int32)
    Es = np.empty(end-start, dtype=np.float64)
    for i, R in enumerate(range(start, end)):
        Rs[i] = R
        Es[i] = E(FIGrad, centrePos[0], centrePos[1], R)
    return Rs, Es


#@jit#(nopython=True, parallel=True)
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




if __name__ == '__main__':
    #pass
    '''
    arr1 = np.array([[1, 2], [1, 2]], dtype='float32')
    arr2 = np.copy(arr1)
    print(mult(arr1, arr1))
    c = circle([10, 10], 5, dtype='complex64', value=255)
    plt.imshow(np.real(c))
    plt.show()
    arr = np.ones([5000, 5000], dtype=np.complex64)
    img = pad(arr, [10000, 10000], [5000, 5000])
    plt.imshow(np.log(np.abs(img)))
    plt.show()
    arr = np.ones([5000, 5000], dtype='float32')
    img = crop(arr, [100, 100])
    plt.imshow(img)
    plt.show()
    '''
    img = cv.imread('../bigC.jpg')[:, :, 0]/255
    out = fft2(np.array(img, dtype=np.complex64))
    start = timer()
    out = fft2(np.array(img, dtype=np.complex64))
    elapsed_time = timer() - start
    print("Time forward + shift: {}".format(elapsed_time))
    #out1 = fftShift(out)
    plt.imshow(np.log(np.abs(out)))
    plt.show()
    out2 = ifft2(out)
    start = timer()
    out2 = ifft2(out)
    elapsed_time = timer() - start
    print("Time inverse + shift: {}".format(elapsed_time))
    plt.imshow(np.real(out2))
    plt.show()
    start = timer()
    out = np.fft.fft2(img) / (img.shape[0] * img.shape[1])
    out = np.fft.fftshift(out)
    elapsed_time = timer() - start
    print("Time np forward + shift: {}".format(elapsed_time))
    plt.imshow(np.log(np.abs(out)))
    plt.show()
    start = timer()
    out = np.fft.ifft2(out)
    elapsed_time = timer() - start
    print("Time numpy inverse: {}".format(elapsed_time))



    '''
    img = cv.imread('../bigC.jpg')[:, :, 0]/255
    start = timer()
    trans = translate(img, 900, 900)
    elapsed_time = timer() - start
    print("Time: {}".format(elapsed_time))
    plt.imshow(trans)
    plt.show()
    start = timer()
    trans = translate(img, 900, 900)
    elapsed_time = timer() - start
    print("Time: {}".format(elapsed_time))
    plt.imshow(trans)
    plt.show()
    start = timer()
    trans = E.transArr(img, 900, 900)
    elapsed_time = timer() - start
    print("Time: {}".format(elapsed_time))
    plt.imshow(np.real(trans))
    plt.show()
    '''


