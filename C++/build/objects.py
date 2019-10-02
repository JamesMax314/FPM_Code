import numpy as np
import cv2 as cv
import methods as meth
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import pandas
import os
import noGPU as h
import matplotlib.pyplot as plt

class fullSys():
    def __init__(self, dir, file, size, line):
        csv_reader = pandas.read_csv(file, index_col='Objective')
        self.Params = {}
        self.Params['mag'] = csv_reader['Magnification'][line]
        self.Params['NA'] = csv_reader['NA'][line]
        self.Params['ps'] = [csv_reader['Pixel Size x'][line], csv_reader['Pixel Size y'][line]]
        self.Params['distance'] = csv_reader['Screen Distance'][line]
        self.Params['LEDSpace'] = csv_reader['LED Spacing'][line]
        self.Params['LEDNum'] = [csv_reader['Num LED x'][line], csv_reader['Num LED x'][line]]
        self.Params['dir'] = dir
        self.Params['images'] = os.listdir(dir)
        self.Params['numImgs'] = len(self.Params['images'])
        self.Params['smallSize'] = meth.readImage(dir, self.Params['images'][0], colour=1, getsize=True)
        self.Params['fResolution'] = self.fRes(self.Params['mag'], self.Params['smallSize'], self.Params['ps'])
        print("fullSys")

        ## Instantiate sub Objects ##

        splitSize, self.Params['lc'] = self.getSS()
        img = meth.readImage(self.Params['dir'], self.Params['images'][0])
        print("fullSys2")

        numFiles, divisor = self.getDivisor(img, splitSize)
        print("fullSys2")

        self.Params['numFiles'] = numFiles
        self.Params['divisor'] = divisor
        self.Params['size'] = self.getSize(size, numFiles)

        self.subObjs = np.empty([numFiles, numFiles], dtype=section)
        print("fullSys1")

        for i in range(numFiles):
            for j in range(numFiles):
                subImg = img[i * divisor:(i + 1) * divisor, j * divisor:(j + 1) * divisor]
                self.subObjs[i, j] = section(i, j, subImg, self.Params)
                h.progbar(i, numFiles, 'Initializing')


    def getSS(self):
        """ Determines the required subsection size based on Cittert Zernike theorem """
        rho = 300e-6 # LED size
        lc = 0.61*R*530/rho
        size = lc*slef.Params['mag'] / self.Params['ps']
        return size, lc


    def getDivisor(self, img, splitSize):
        imgSize = img.shape[0]
        while True:
            if imgSize % splitSize == 0:
                divisor = splitSize
                break
            splitSize += 1
        numFiles = int(imgSize / divisor)
        return numFiles, divisor


    def getSize(self, size, numSplits):
        while True:
            if size[0] % numSplits == 0:
                break
            size[0] += 1
        return size[0]


    def fRes(self, mag, size, ps):
        """ Determines the change in spatial frequency across one pixel in F-space """
        x = 2 * np.pi * mag / (size[0] * ps[0])
        y = 2 * np.pi * mag / (size[1] * ps[1])
        return [x, y]


class section():
    def __init__(self, i0, j0, subImg, Params):
        self.Params = Params
        self.subParams = {}
        self.subParams['wLen'] = [630e-9, 530e-9, 430e-9]
        self.subParams['subSize'] = subImg.shape
        self.subParams['bigSize'] = [np.int(Params['size'] / Params['numFiles'])] * 2
        self.S = np.empty([self.subParams['bigSize'][0], self.subParams['bigSize'][1], 3], dtype=np.complex64)
        self.P = np.empty([self.subParams['subSize'][0], self.subParams['subSize'][1], 3], dtype=np.complex64)
        self.meanFFT = np.zeros([self.subParams['subSize'][0], self.subParams['subSize'][1], 3], dtype=np.complex64)
        self.meanNum = 0
        self.subParams['fRApprox'] = np.empty([3], dtype=int)
        self.subParams['coords'] = np.empty([3, 16, 16, 2])
        self.subParams['isBF'] = np.empty([3, 16, 16])
        for i in range(0, 3):
            self.S[:, :, i] = self.initS0(subImg[:, :, i], self.subParams['bigSize'])
            self.subParams['fRApprox'][i] = self.fRad(Params['fResolution'],
                                                      Params['NA'], self.subParams['wLen'][i])
            print(Params['NA'], self.subParams['wLen'][i], Params['mag'], Params['ps'], Params['smallSize'])
            self.P[:, :, i] = self.initP0(self.subParams['subSize'], self.subParams['fRApprox'][i])
            self.subParams['coords'][i, :, :, :], self.subParams['isBF'][i, :, :] =\
                self.initCoords(i0, j0, self.subParams['wLen'][i], self.subParams['fRApprox'][i])
        self.bayer = np.empty([Params['divisor'], Params['divisor'], 3])
        self.invBayer = np.empty([Params['divisor'], Params['divisor'], 3])
        for i in range(3):
            self.bayer[:, :, i], self.invBayer[:, :, i] = h.genBayer([Params['divisor'], Params['divisor']], i)


    def initS0(self, img, size):
        """ Initialises the FT of the high res image by linear interpolation of a low res image """

        I0 = cv.resize(img, (size[1], size[0]),
                       interpolation=cv.INTER_LINEAR)  # Bilinear interpolated upsampled image

        amplitude = np.sqrt(I0)

        FI0 = fft2(ifftshift(amplitude))
        FI0 = fftshift(FI0) # FI0.shape[0]
        S = np.array(FI0, dtype=np.complex64)
        return S


    def initP0(self, size, radius):
        """ Initialises the pupil function as a real circular step function of value 1 """
        return h.circle(size, radius)[:, :, 0]


    def fRad(self, fDu, NA, wLen):
        """ Determines the approximate radius in F-space in pixels of the pupil function """
        x = 2 * np.pi * NA / (wLen * fDu[0])
        y = 2 * np.pi * NA / (wLen * fDu[1])
        avr = np.int32(np.average([x, y]))
        return avr


    def initCoords(self, i, j, wLen, Rad):
        """ Returns 2D array where LED coords relate to fourier centre positions """
        segmentPos = [i, j]
        n = self.Params['numFiles']
        w = self.subParams['subSize'][0]
        c = w / (2 * n)
        centre = (segmentPos[0] * 2 * c + c - w) * self.Params['ps'][0]/self.Params['mag']
        self.Params['centre'] = centre
        coords = np.empty((self.Params['LEDNum'][0], self.Params['LEDNum'][1], 2), dtype=np.int32)
        isBF = np.zeros((self.Params['LEDNum'][0], self.Params['LEDNum'][1]), dtype=np.int32)
        numImgs = int(len(self.Params['images']) ** 0.5)
        for i, img in enumerate(self.Params['images']):
            LED = meth.getLED(img)
            LEDPixelPos = self.getLEDPos(LED[0], LED[1], centre, wLen)
            #print("LED:", LED, "LEDPixelPos:", LEDPixelPos)
            #print("LEDPos:", [LED[0] + int(numImgs / 2) - 1, LED[1] + int(numImgs / 2) - 1])
            coords[LED[0] + int(numImgs / 2) - 1, LED[1] + int(numImgs / 2) - 1] = LEDPixelPos
            if ((LEDPixelPos[0]-w/2)**2 + (LEDPixelPos[1]-w/2)**2 < Rad):
                isBF[LED[0] + int(numImgs / 2) - 1, LED[1] + int(numImgs / 2) - 1] = 1
        return coords, isBF


    def getLEDPos(self, nx, ny, centre, wLen):
        """ Determines the location of the centre of the fourier pattern in pixels """
        ax = np.arctan((centre - nx * self.Params['LEDSpace']) / self.Params['distance'])  # Angle to x axis
        ay = np.arctan((centre - ny * self.Params['LEDSpace']) / self.Params['distance'])  # Angle to y axis
        dx = ax / (wLen * self.Params['fResolution'][0])
        dy = ay / (wLen * self.Params['fResolution'][1])
        pos = [int(dx + self.subParams['subSize'][0] / 2), int(dy + self.subParams['subSize'][0] / 2)]
        return pos


class splitImage():
    def __init__(self, dir, imgName, numSplits, splitSize):
        self.LEDPos = meth.getLED(imgName)
        self.subImg = np.empty([numSplits, numSplits], dtype=subImage)
        for i in range(numSplits):
            for j in range(numSplits):
                self.subImg[i, j] = subImage(dir, splitSize, imgName, self.LEDPos, i, j)


class subImage():
    def __init__(self, dir, splitSize, imgName, LEDPos, i, j):
        img = meth.readImage(dir, imgName)
        self.image = img[i * splitSize:(i + 1) * splitSize, j * splitSize:(j + 1) * splitSize]
        self.imgPos = [i, j]
        self.LEDPos = LEDPos






########################################################################################################################
'''
class preProcess(objective):
    def __init__(self, dir, file, size, line, colour=1):
        """ Slices images into sections """
        super().__init__(dir, file, size, line, colour=1)
        numFiles, devisor = self.getDevisor(150)
        self.genFiles(numFiles)
        self.split(devisor, numFiles)


    def genFiles(self, numFiles):
        path = os.path.join(os.getcwd(), 'temp')
        if os.path.isdir(path):
            shutil.rmtree(path)
        time.sleep(0.01)
        os.mkdir(path)
        for i in range(numFiles):
            for j in range(numFiles):
                folder = '%s_%s' % (str(i), str(j))
                path1 = os.path.join(path, folder)
                os.mkdir(path1)


    def getDevisor(self, splitSize):
        imgName  = self.images[0]
        img = self.readImage(self.dir, imgName)
        imgSize = img.shape[0]
        while True:
            if imgSize % splitSize == 0:
                devisor = splitSize
                break
            splitSize += 1
        numFiles = int(imgSize / devisor)
        return numFiles, devisor


    def split(self, devisor, numFiles):
        path0 = os.path.join(os.getcwd(), 'temp')
        for i0, file in enumerate(self.images):
            LED = self.getLED(file)
            img = self.readImage(self.dir, file)
            for i in range(numFiles):
                for j in range(numFiles):
                    folder = '%s_%s' % (str(i), str(j))
                    path1 = os.path.join(path0, folder)
                    file = 'img_%s_%s_.jpg' % (str(LED[0]), str(LED[1]))
                    path = os.path.join(path1, file)
                    subImg = img[i * devisor:(i + 1) * devisor, j * devisor:(j + 1) * devisor]
                    cv.imwrite(path, subImg)
                    h.progbar(i0 * numFiles ** 2 + i * numFiles + j,
                              len(self.images) * numFiles ** 2, 'Slicing Images')



    def initCoords(self, dir):
        """ Returns 2D array where LED coords relate to fourier centre positions """
        dirName = os.path.basename(dir)
        segmentPos = self.getSegment(dirName)
        N = len(os.listdir(dir))
        n = np.sqrt(N)
        w = self.smallSize[0]
        c = w / (2 * n)
        centre = (segmentPos[0] * 2 * c + c - w) * self.ps[0]/self.mag
        coords = np.empty((self.LEDNum[0], self.LEDNum[1], 2), dtype=np.int32)
        for i, img in enumerate(self.images):
            LED = self.getLED(img)
            LEDPixelPos = self.getLEDPos(LED[0], LED[1], centre)
            coords[LED[0], LED[1]] = LEDPixelPos
        return coords
'''