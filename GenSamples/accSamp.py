import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import noGPU
import hsHelper_V3 as h
from numpy.fft import fft2, fftshift, ifft2, ifftshift

##### Meta Data #####
ledSpace = 5e-4
ledSep = 4e-2
pixSize = 1.12e-6
wls = [630e-9, 530e-9, 430e-9]
M = 750 # Sensor size
NA = 0.25
mag = 2
inSize = 1e-2
R = np.empty([3])
for i in range(3):
    R[i] = M*pixSize*NA/(wls[i]*mag)

##### get input file #####
file = "sample1.jpg"
img = cv.imread(file)
img = noGPU.crop(img, [720, 720])
inPixSize = inSize/720

pupil = np.empty([img.shape[0], img.shape[1], img.shape[2]])
for i in range(3):
    pupil[:, :, i] = noGPU.circle([720, 720], R[i])[:, :, 0]/255

def getPhase(i, j):
    global ledSpace, ledSep, inPixSize
    thetaX = np.arctan(i*ledSpace/ledSep)
    thetaY = np.arctan(j*ledSpace/ledSep)
    basePhaseStepX = 2*np.pi*inPixSize*np.sin(thetaX)/wls
    basePhaseStepY = 2*np.pi*inPixSize*np.sin(thetaY)/wls
    return basePhaseStepX, basePhaseStepY


def fillPhase(xs, ys, stepX, stepY):
    out = np.sqrt((xs*stepX)**2 + (ys*stepY)**2)
    return out % (2*np.pi)


def circPhase(xs, ys, i, j, c):
    global ledSpace, ledSep, inPixSize, wls
    x = xs-np.amax(xs)/2
    y = ys-np.amax(ys)/2
    rad = np.sqrt(ledSep**2) # (ledSpace*i)**2 + (ledSpace*j)**2 +
    partialRad = np.sqrt((inPixSize*x - i*ledSpace)**2 + (inPixSize*y - j*ledSpace)**2 + ledSep**2)
    phase = (rad-partialRad)*2*np.pi/wls[c] % (2*np.pi)
    return phase


for i in range(-7, 8):
    for j in range(-7, 8):
        phaseSteps = getPhase(i, j)
        phase = np.ones([img.shape[0], img.shape[1], img.shape[2]])
        y = np.linspace(0, int(img.shape[0]), img.shape[0])
        x = np.linspace(0, int(img.shape[1]), img.shape[1])
        xx, yy = np.meshgrid(x, y)
        fft = np.empty([img.shape[0], img.shape[1], img.shape[2]], dtype=np.complex64)
        ifft = np.empty([img.shape[0], img.shape[1], img.shape[2]], dtype=np.complex64)
        for c in range(3):
            #phase[:, :, c] = fillPhase(xx, yy, phaseSteps[0][c], phaseSteps[1][c])
            phase[:, :, c] = circPhase(xx, yy, i, j, c)
            #pass
        #plt.imshow(phase/np.amax(phase))
        #plt.show()
        for c in range(3):
            img1 = np.array(img*np.exp(1j*phase))
            fft[:, :, c] = fftshift(fft2(ifftshift(img1[:, :, c])))
            thetaX = np.arctan(i * ledSpace / ledSep)*(M*inPixSize/(wls[c]*2*np.pi*mag))
            thetaY = np.arctan(j * ledSpace / ledSep)*(M*inPixSize/(wls[c]*2*np.pi*mag))
            fftTrans = h.translate(fft[:, :, c], -thetaX, -thetaY)
            fft[:, :, c] = pupil[:, :, c]*fftTrans
            ifft[:, :, c] = fftshift(ifft2(fftshift(fft[:, :, c])))
        #plt.imshow(np.log(np.abs(fft))[:, :, 0])
        #plt.show()
        #plt.imshow(np.array(np.abs(ifft)*255, dtype=np.int64))
        #plt.show()
        imgName = "./Imgs/img_" + str(i) + "_" + str(j) + "_.jpg"
        cv.imwrite(imgName, np.array(np.abs(ifft)*255, dtype=np.int64))
        h.progbar((i+7)*16+(j+7), 16**2, "Generating")
