import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import objects
import matplotlib.pyplot as plt
import noGPU as h
import methods as meth
import cv2 as cv
import pickle
import time
import callibrate_V1 as cal
#import pyTest


def updateS_1(P, PHI, PSI, coords):
    """ Returns the update step for S Note that the fitting parameter alpha is not included """
    DELTA = PHI - PSI
    DELTATrans = h.translate(DELTA, -coords[0], -coords[1])
    PTrans = h.translate(P, -coords[0], -coords[1])
    numerator = np.conj(PTrans) * DELTATrans
    denominator = np.amax(np.abs(PTrans)) ** 2
    frac = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return frac


def updateP_1(S, PHI, PSI, coords):
    """ Returns the update step for S Note that the fitting parameter alpha is not included """
    DELTA = PHI - PSI
    STrans = h.translate(S, coords[0], coords[1])
    numerator = np.conj(STrans) * DELTA  # np.abs(STrans) * np.conj(STrans) * DELTATrans
    denominator = np.amax(np.abs(STrans)) ** 2  # np.amax(np.abs(S)) * (np.abs(STrans) ** 2 * delta)
    frac = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return frac


def FPM(section, subImg):
    col = [2, 1, 0]
    for ind in range(3):#, i in enumerate(col):
        i = col[ind]
        img = subImg.image[:, :, i]
        S = section.S[:, :, i]
        P = section.P[:, :, i]
        bayer = section.bayer[:, :, ind]
        #print("Bayer: ", bayer[0, 0], bayer[0, 1], bayer[1, 0], bayer[1, 1])
        invBayer = section.invBayer[:, :, ind]
        #print("Bayer: ", invBayer[0, 0], invBayer[0, 1], invBayer[1, 0], invBayer[1, 1])
        LED = subImg.LEDPos

        dx, dy = section.subParams['coords'][ind, LED[0], LED[1], 0] - section.subParams['subSize'][0] / 2, \
                 section.subParams['coords'][ind, LED[0], LED[1], 1] - section.subParams['subSize'][0] / 2

        #print("LED:", LED[0], LED[1])
        #print("dx, dy:", dx, dy)
        #print("dx:", section.subParams['coords'][i, LED[0], LED[1], 0], section.subParams['coords'][i, LED[0], LED[1], 1])

        STrans = h.translate(S, dx, dy)
        SCrop = h.crop(STrans, section.subParams['subSize'])
        phi = P * SCrop
        #cv.imshow("phi", np.abs(phi)/1000)
        PHI = ifftshift(ifft2(fftshift(phi))) #/ phi.shape[0]
        cv.imshow("S", np.abs(S)/1000)

        amplitude = np.sqrt(img * bayer)
        bayerCorrectedAmp = amplitude + np.abs(PHI * invBayer)

        PPNum = bayerCorrectedAmp * PHI
        PPDen = np.abs(PHI)
        PHIPrimed = np.divide(PPNum, PPDen,  out=np.zeros_like(PPNum), where=PPDen != 0)

        phiPrimedPartial = fft2(ifftshift(PHIPrimed))
        phiPrimed = fftshift(phiPrimedPartial) #/ phiPrimedPartial.shape[0]

        #plt.imshow(np.log(np.abs(phiPrimed)))
        #plt.show()

        #print("max var:", np.amax(np.abs(phi)))

        updateStepS = updateS_1(P, phiPrimed, phi, [dx, dy])
        S = S + h.pad(updateStepS, section.subParams['bigSize'])
        #print("max phi:", np.amax(np.abs(phi)))

        updateStepP = updateP_1(h.crop(S, section.subParams['subSize']), phiPrimed, phi, [dx, dy])
        P = P + updateStepP

        section.S[:, :, i] = S
        section.P[:, :, i] = P
        #print("Max uS:", np.amax(np.abs(updateStepS)))
        #cv.imshow("phiPrimed", np.abs(phiPrimed)/1000)
        #cv.imshow("img", np.abs(ifftshift(fft2(fftshift(section.S[:, :, 2]))))/1000000)
        #cv.waitKey(0)



def FPM_Feed(FPMSys):
    """ Performs phase reconstruction following a gradient decent implementation of Girchberg-Sax """
    dir = FPMSys.Params['dir']
    images = FPMSys.Params['images']
    numSplits = FPMSys.Params['numFiles']
    divisor = FPMSys.Params['divisor']

    imgIndex = arrangeImages(images)

    tStart = time.time()
    print('')
    for i0 in range(len(images)):
        imgName = images[i0]#imgIndex[i0]]
        subImgs = objects.splitImage(dir, imgName, numSplits, divisor)

        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)

        for i in range(FPMSys.Params['numFiles']):
            for j in range(FPMSys.Params['numFiles']):
                section = FPMSys.subObjs[i, j]
                subImg = subImgs.subImg[i, j]
                #t = time.time()
                FPM(section, subImg)
                #plt.imshow(np.abs(section.S[:, :, 0]))
                #plt.show()
                #print("main: ", time.time()-t, "s")

                h.progbar_time(i0*(FPMSys.Params['numFiles']**2) + i*FPMSys.Params['numFiles'] + j,
                          len(images)*(FPMSys.Params['numFiles']**2), 'Main FPM', tStart)
                
    S = FPMSys.subObjs[0, 0].S
    out = np.empty_like(S)
    for i in range(3):
        out[:, :, i] = ifftshift(ifft2(fftshift(S[:, :, i])))
    plt.imshow(np.abs(out) / np.amax(np.abs(out)))
    plt.show()
    P = FPMSys.subObjs[0, 0].P
    out = np.empty_like(P)
    for i in range(3):
        out[:, :, i] = ifftshift(ifft2(fftshift(P[:, :, i])))
    plt.imshow(np.abs(out) / np.amax(np.abs(out)))
    plt.show()


def callib_Feed(FPMSys):
    dir = FPMSys.Params['dir']
    images = FPMSys.Params['images']
    numSplits = FPMSys.Params['numFiles']
    divisor = FPMSys.Params['divisor']
    tStart = time.time()
    print('')
    for i0, img in enumerate(images):
        subImgs = objects.splitImage(dir, img, numSplits, divisor)
        for i in range(FPMSys.Params['numFiles']):
            for j in range(FPMSys.Params['numFiles']):
                section = FPMSys.subObjs[i, j]
                subImg = subImgs.subImg[i, j]
                cal.meanFFT(section, subImg)

                h.progbar_time(i0 * (FPMSys.Params['numFiles'] ** 2) + i * FPMSys.Params['numFiles'] + j,
                               len(images) * (FPMSys.Params['numFiles'] ** 2), 'MeanFFT', tStart)
    plt.imshow(np.abs(FPMSys.subObjs[0, 0].meanFFT)/np.amax(np.abs(FPMSys.subObjs[0, 0].meanFFT)))
    plt.show()


def arrangeImages(imgNames):
    """ Returns an array of indices for image files sorted based on the LED offset """
    indices = [0] * len(imgNames)
    radius = [0] * len(imgNames)
    for i, iName in enumerate(imgNames):
        LED = meth.getLED(iName)
        radius[i] = LED[0] ** 2 + LED[1] ** 2
        indices[i] = i
    order = [x for _, x in sorted(zip(radius, indices))]
    return order


def main(dir, file, size, line):
    FPMSys = objects.fullSys(dir, file, size, line)
    print("MaxS:", np.amax(np.abs(FPMSys.subObjs[0, 0].S[:, :, 0])))

    #plt.imshow(np.abs(FPMSys.subObjs[0].P0))
    #plt.show()
    #pickle.dump(FPMSys, open("save.p", "wb"))
    #FPMSys = pickle.load(open("save.p", "rb"))
    #split = objects.splitImage(dir, 'img_0_0_.jpg', FPMSys.Params['numFiles'], FPMSys.Params['divisor'])
    #plt.imshow(split.subImg[0, 0].image/255)
    #plt.show()
    #callib_Feed(FPMSys)
    FPM_Feed(FPMSys)



if __name__ == '__main__':
    dir = '..\Images'
    file = '..\Params\Objective.txt'
    size = np.array([400, 400])
    line = 1
    main(dir, file, size, line)