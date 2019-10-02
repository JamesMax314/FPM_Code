import cv2 as cv
import numpy as np
import test3 as c
import matplotlib.pyplot as plt
import time

def comp(n, m):
    arr = np.empty([n, m], dtype=np.complex64)
    count = 1
    count1 = -1
    for i in range(n):
        for j in range(m):
            arr[i, j] = count + 1j * count1
            count += 1
            count1 -= 1

    print(arr)
    out = c.test(arr, arr, n)

def real(n, m):
    arr = np.empty([n, m, 3])
    count = 1
    for i in range(n):
        for j in range(m):
            for k in range(3):
                arr[i, j, k] = count
                count += 1

    print(arr)
    out = c.test(arr, arr, n)

def imgT(imgN):
    img = cv.imread(imgN)#[:, :, 0]
    img = np.array(img)
    start = time.clock()
    red = img[:, :, 2]
    print("Python: " + str(time.clock()-start) + "seconds")
    out = c.test(img, img)
    plt.imshow(np.real(out))
    plt.show()

def casTest():
    arr = np.ones([3,16,16,3])
    c.test1(arr)

def testFft():
    arr = cv.imread("sample1.jpg")[:, :, 0]
    out = c.fft(np.array(arr, dtype=np.complex64))
    plt.imshow(np.log(np.abs(out)))
    plt.show()
    out1 = c.ifft(out)
    plt.imshow(out1)
    plt.show()

def conj():
    arr = cv.imread("sample1.jpg")[:, :, 0]
    arr = np.array(arr*1j+arr, np.complex64)
    print(arr)
    out1 = c.conjTest(arr)
    print(out1)

if __name__ == "__main__":
    #comp(4, 4)
    #real(5, 5)
    #imgT("sample1.jpg")
    #casTest()
    #testFft()
    conj()