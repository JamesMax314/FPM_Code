import cv2 as cv
import numpy as np
from PIL import Image
import noGPU as h
import pandas
import os

def readImage(dir, file, colour=-1, getsize=False, crop=True):
    """ Reads an image file to an array kwargs: getsize returns only the smallest dimension of image
    do not specify colour for monochrome images """
    path = os.path.join(dir, file)
    img = cv.imread(path)

    if colour >= 0:
        img = img[:, :, colour]

    inSize = np.array(img.shape)
    smallSize = np.amin(inSize[0:1])

    if getsize:
        del img
        return [smallSize, smallSize]

    if crop:
        imgOut = h.crop(img, [smallSize, smallSize])
        return imgOut

    return img


def getLED(imgName):
    """ Returns the location of LED in matrix """
    arrCSV = imgName.split('_')
    return [int(arrCSV[1]), int(arrCSV[2])]