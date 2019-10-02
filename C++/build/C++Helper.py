import numpy as np
import cFPM as c
import objects
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import cv2 as cv
import noGPU as h
import time
import callibrate_V1 as cal
import mainFPM_V1 as ma
import os
import pickle as pkl


def updateS(fullSys, cFullSys):
    for i in range(fullSys.Params["numFiles"]):
        for j in range(fullSys.Params["numFiles"]):
            sub = fullSys.subObjs[i, j]
            index = np.array([i, j])
            for k in range(3):
                sub.P[:, :, k] = cFullSys.export_P(index, k)


def updateS(fullSys, cFullSys):
    for i in range(fullSys.Params["numFiles"]):
        for j in range(fullSys.Params["numFiles"]):
            sub = fullSys.subObjs[i, j]
            index = np.array([i, j])
            for k in range(3):
                sub.S[:, :, k] = cFullSys.export_S(index, k)

def updatemaenFFT(fullSys, cFullSys):
    for i in range(fullSys.Params["numFiles"]):
        for j in range(fullSys.Params["numFiles"]):
            sub = fullSys.subObjs[i, j]
            index = np.array([i, j])
            for k in range(3):
                sub.meanFFT[:, :, k] = cFullSys.export_meanFFT(index, k)

def updateMeanFFT(fullSys, cFullSys):
    for i in range(fullSys.Params["numFiles"]):
        for j in range(fullSys.Params["numFiles"]):
            sub = fullSys.subObjs[i, j]
            index = np.array([i, j])
            for k in range(3):
                sub.meanFFT[:, :, k] = cFullSys.export_meanFFT(index, k)


def passFullSys(fullSys):
    print("Ok")
    #print("py divisor: ", fullSys.Params["divisor"])
    cFullSys = c.fullSys()
    c.fill_dict_str(cFullSys, fullSys.Params)
    cFullSys.add_images(fullSys.Params["images"])
    #print(fullSys.Params["images"])
    #with open("..//tmp//subObj.pkl", "rb") as f:
    #   fullSys.subObjs = pkl.load(f)
    #cFullSys.list()
    print("Ok")
    for i in range(fullSys.Params["numFiles"]):
        for j in range(fullSys.Params["numFiles"]):
            sub = fullSys.subObjs[i, j]
            #print(sub.subParams["coords"])
            #plt.imshow(np.abs(np.fft.fftshift(np.fft.ifft2(sub.S[:, :, 0]))))
            #plt.show()
            subSys = c.subSys()
            #plt.imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(sub.S[:, :, 0]))))
            #plt.show()
            #img = cv.imread("..//Images//img_0_0_.jpg")
            #img1 = np.array(np.fft.fftshift(np.fft.fft2(img[:, :, 0])), dtype=np.complex64)
            #out = c.fft(img1)
            #out1 = c.fft(fullSys.subObjs[i, j].S[:, :, 0])
            #plt.imshow(np.abs(np.fft.fftshift(out)))
            #plt.show()
            s_0 = np.ascontiguousarray(sub.S[:, :, 0], dtype=np.complex64)
            s_1 = np.ascontiguousarray(sub.S[:, :, 1], dtype=np.complex64)
            s_2 = np.ascontiguousarray(sub.S[:, :, 2], dtype=np.complex64)
            #print("s: ", str(s_0[200, 200]))
            subSys.add_S(s_0, s_1, s_2)
            p_0 = np.ascontiguousarray(sub.P[:, :, 0], dtype=np.complex64)
            p_1 = np.ascontiguousarray(sub.P[:, :, 1], dtype=np.complex64)
            p_2 = np.ascontiguousarray(sub.P[:, :, 2], dtype=np.complex64)
            subSys.add_P(p_0, p_1, p_2)
            #plt.imshow(np.log(np.abs(sub.meanFFT[:, :, 0])))
            #plt.show()
            m_0 = np.ascontiguousarray(sub.meanFFT[:, :, 0], dtype=np.complex64)
            m_1 = np.ascontiguousarray(sub.meanFFT[:, :, 1], dtype=np.complex64)
            m_2 = np.ascontiguousarray(sub.meanFFT[:, :, 2], dtype=np.complex64)
            subSys.add_meanFFT(m_0, m_1, m_2)
            #subSys.add_meanFFT(sub.meanFFT[:, :, 0], sub.meanFFT[:, :, 1], sub.meanFFT[:, :, 2])
            subSys.add_wLen(sub.subParams["wLen"])
            rad = np.ascontiguousarray(sub.subParams["fRApprox"])
            subSys.add_fRApprox(rad)
            subSys.add_coords(sub.subParams["coords"])
            BF = np.ascontiguousarray(sub.subParams["isBF"])
            subSys.add_isBF(BF)
            subSys.add_subSize(sub.subParams["subSize"])
            subSys.add_bigSize(sub.subParams["bigSize"])
            subSys.add_bayer(sub.bayer)
            subSys.add_invBayer(sub.invBayer)
            cFullSys.add_subSys(subSys, i, j)
    print("Ok")
    #c.meanFFT(cFullSys)
    #updateMeanFFT(fullSys, cFullSys)
    #with open("..//tmp//subObj.pkl", "wb") as f:
    #    pkl.dump(fullSys.subObjs, f)
    #print("ok")
    #return 0
    c.BrightField(cFullSys)
    print("Ok")
    #print(fullSys.subObjs[0, 0].meanFFT[:, :, 0])
    plt.imshow(np.abs(fullSys.subObjs[0, 0].meanFFT[:, :, 0]))
    plt.show()
    c.FPM_Feed(cFullSys)
    #plt.imshow(np.abs(out))
    #plt.show()
    updateS(fullSys, cFullSys)
    plt.imshow(np.abs(fullSys.subObjs[0, 0].S) / np.amax(np.abs(fullSys.subObjs[0, 0].S)))
    plt.show()
    S = fullSys.subObjs[0, 0].S
    out = np.empty_like(S)
    for i in range(3):
        out[:, :, i] = ifftshift(ifft2(fftshift(S[:, :, i])))
    plt.imshow(np.abs(out) / np.amax(np.abs(out)))
    plt.show()
    plt.imshow(np.abs(fullSys.subObjs[0, 0].P) / np.amax(np.abs(fullSys.subObjs[0, 0].P)))
    plt.show()


def main(dir, file, size, line):
    ma.callib_Feed(fullSys)
    FPMSys = objects.fullSys(dir, file, size, line)
    print("Ok")
    passFullSys(FPMSys)

def test():
    img = cv.imread("..//Images//img_0_0_.jpg")
    plt.imshow(img)
    plt.show()
    img1 = np.array(img[:, :, 0], dtype=np.complex64)
    out = c.fft(img1)
    plt.imshow(np.abs(out))
    plt.show()
    out1 = np.fft.fftshift(np.fft.fft2(img1))
    plt.imshow(np.abs(out1))
    plt.show()

def testSqrt():
    img = cv.imread("..\Images\img_0_0_.jpg")[:, :, 0]
    plt.imshow(img)
    plt.show()
    outPyth = np.array(np.sqrt(img), dtype=np.float32)
    plt.imshow(outPyth)
    plt.show()
    img = np.ascontiguousarray(img)
    out = c.sqrtTest(img)
    plt.imshow(out)
    plt.show()

def abTest():
    img = cv.imread("..\Images\img_0_0_.jpg")[:, :, 0]
    arr = np.array(1j*img, dtype=np.complex64)
    #z = np.zeros_like(img, dtype=np.complex64)
    out = c.absTest(arr)
    plt.imshow(np.imag(out))
    plt.show()

def t1():
    img = cv.imread("..\Images\img_0_0_.jpg")[:, :, 0]
    arr = np.array(img, dtype=np.complex64)
    out = c.fft(arr)
    print("max:", np.amax(np.abs(out)))
    outp = np.fft.fft2(arr)
    print("maxp:", np.amax(np.abs(outp)))
    out1 = c.ifft(arr)
    print("max:", np.amax(np.abs(out1)))
    outp1 = np.fft.ifft2(arr)
    print("maxp:", np.amax(np.abs(outp1)))

def t2():
    img = cv.imread("..\Images\img_0_0_.jpg")[:, :, 0]
    arr = np.array(img, dtype=np.complex64)
    out = c.fft(arr)
    plt.imshow(np.abs(arr))
    plt.show()

def t3():
    img = cv.imread("..\Images\img_0_0_.jpg")#[:, :, 0]
    out = c.testTrans(img, 100, 0)
    plt.imshow(out)
    plt.show()
    img1 = np.array(img, dtype=np.complex64)
    out1 = h.translate(img1, 100, 0)
    plt.imshow(np.abs(out1)/255)
    plt.show()

def t4():
    img = "..\Images\img_0_0_.jpg"
    c.testRead(img)

def t5():
    arr = np.ones([400, 400], dtype=np.complex64)*10*1j
    arr1 = np.ones([50, 50], dtype=np.complex64)
    arr2 = np.zeros([400, 400], dtype=np.complex64)
    arr2[0:50, 0:50] = arr1*50 + 1j*arr1*60
    arr2 = np.ascontiguousarray(arr2)
    arr = np.ascontiguousarray(arr)
    out = c.testDiv(arr2, arr)
    plt.imshow(np.imag(out))
    plt.show()
    out2 = c.conjTest(arr)
    plt.imshow(np.imag(out2))
    plt.show()
    out2 = c.absTest(arr)
    plt.imshow(np.imag(out2))
    plt.show()
    start = time.time()
    out1 = np.divide(arr2, arr)
    end = time.time()
    print("dividePy:", end - start)
    plt.imshow(np.imag(out1))
    plt.show()



if __name__ == "__main__":
    #testSqrt()
    #abTest()
    #t5()
    print("Ok")
    dir = '..\Images'
    file = '..\Params\Objective.txt'
    size = np.array([4000, 4000])
    line = 7
    main(dir, file, size, line)

