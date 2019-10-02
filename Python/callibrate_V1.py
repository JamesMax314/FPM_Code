""" This file contains functions used to callibrate an FPM setup """
import noGPU as h
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import methods as meth

def meanFFT(section, subImg):
    """ Returns the mean value of FFT accross all images used to remove DC offset """
    mean = section.meanFFT
    num = section.meanNum
    for i in range(3):
        img = subImg.image[:, :, i]
        FIAbs = np.abs(fft2(ifftshift(img)))
        FIAbsShift = fftshift(FIAbs)  # / FIAbs.shape[0]**2
        mean[:, :, i] = mean[:, :, i] * num + FIAbsShift
    num += 1
    section.meanFFT = mean / num
    section.meanNum = num


def brightField(colour, brightField, meanF):
    """ Bright field LED position calibration """
    brightFieldImg = getBrightField(brightField)

    newCoords = np.empty((len(brightFieldImg), 2))
    oldCoords = np.empty((len(brightFieldImg), 2))
    LEDPos = np.empty((len(brightFieldImg), 2))

    for i, img in enumerate(brightFieldImg):
        imgArr = self.readImage(self.dir, img, colour=colour)
        LED = self.getLED(img)

        oldCoords[i] = self.coords[LED[0], LED[1]]
        newCoords[i] = self.circEdge(imgArr, img, self.fRApprox, oldCoords[i], meanF)

        LEDPos[i] = [LED[0], LED[1]]

        h.progbar(i, len(brightFieldImg), 'Bright Field calibration')

    matrix, inliers = h.RANSACTransform(oldCoords, newCoords)

    for i, imgFile in enumerate(self.images):
        LED = self.getLED(imgFile)

        if h.isInArr(LEDPos, LED):
            index = h.whereArr(LEDPos, LED)[0]

            if inliers[index] == True:
                self.coords[LED[0], LED[1]] = newCoords[index]
            else:
                self.coords[LED[0], LED[1]] = h.transMatrix(newCoords[index], matrix)
        else:
            self.coords[LED[0], LED[1]] = h.transMatrix(self.coords[LED[0], LED[1]], matrix)


def circEdge(self, arr, imgName, R, coords, meanF):
    """ Identifies the centre of a circle by detecting its edge """
    img = self.prepEdge(arr, meanF, deriv=1)
    centre = h.lineE(img, R, coords)
    std = h.getStd(img, R, centre)
    potentialsE1 = h.getHigh(centre, std)

    intersect = potentialsE1

    rImg = self.readImage(self.dir, imgName, colour=1)
    errorMetric = np.empty(len(intersect))
    for i, coord in enumerate(intersect):
        errorMetric[i] = self.eMetricCentreFit(rImg, img, self.PConstr, coord, self.smallSize)

    minimiser = np.int32(np.argmin(errorMetric))
    bestCentre = intersect[minimiser]
    return bestCentre


def calibrateR(self, brightField, sampleSize, colour, meanF):
    """ Returns the average radius that maximises E1 """
    sampleImages = []
    bFImages = self.getBrightField(brightField)
    i = 0

    while len(sampleImages) < sampleSize:
        random = np.random.randint(0, len(bFImages))
        imgFile = bFImages[random]
        sampleImages = np.append(sampleImages, imgFile)

    R = 0
    for index, i in enumerate(sampleImages):
        img = self.readImage(self.dir, i, colour=colour)

        FIGrad = self.prepEdge(img, meanF)

        LED = self.getLED(i)
        start = int(self.fRApprox) - self.fRApprox // 2
        end = int(self.fRApprox) + np.ceil(self.fRApprox / 2)
        centrePos = self.coords[LED[0], LED[1]]

        Rs, Es = h.checkR(np.int32(start), np.int32(end), np.int32(centrePos), np.float64(FIGrad))
        R += Rs[np.argmax(Es)]

        h.progbar(index, len(sampleImages), 'R Calibration')

    R = int(R / sampleSize)
    return R