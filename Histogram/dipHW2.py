import cv2
import numpy as np
import matplotlib.pyplot as plt

def split(img):
    retR = implement(img[:,:,0])
    retG = implement(img[:,:,1])
    retB = implement(img[:,:,2])
    ret = np.dstack((retR,retG,retB))
    #original image is modified because pointer transfer
    #cv2.imwrite('newColor.png', ret)
    return ret

def implement(imgGray):
    #smaller Image fasten running
    #imgGray = cv2.resize(imgGray, (0, 0), fx = 0.1, fy = 0.1)
    [M,N] = imgGray.shape
    oneDimgGray = [element for row in imgGray for element in row]
    pixelNumbers = [0] * 256
    for i in range(len(oneDimgGray)):
        pixelNumbers[oneDimgGray[i]] += 1
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ##get transform
    ElementNumber = M*N
    pixelProbability = [a/(M*N) for a in pixelNumbers]
    transform = [0]*256
    for k in range(len(transform)):
        temp = sum([pixelProbability[j] for j in range(0,k+1)])
        transform[k] = (256-1)*(temp)
    print(transform)
    oneDtrans = [transform[a] for a in oneDimgGray]
    [n1,bins1,patches1]=ax1.hist(oneDimgGray,30,density=True)
    [n2,bins2,patches2]=ax2.hist(oneDtrans,30,density=True)
    for i in range(len(imgGray)):
        for j in range(len(imgGray[0])):
            imgGray[i][j] = transform[imgGray[i][j]]
    return imgGray
    
    #equ = cv2.equalizeHist(imgGray)
    #res = np.hstack((imgGray,equ)) #stacking images side-by-side
    #cv2.imwrite('res.png',res)
    #cv2.imwrite('trans.jpg', imgGray)
    #fig.savefig('abc.png')
    #print(n)
    #cv2.imwrite('Gray.jpg', imgGray)

def CallLibrary(img): 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(imgGray)
    #res = np.hstack((imgGray,equ)) #stacking images side-by-side
    return equ

def CLAHE(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(24,24))
    cl1 = clahe.apply(imgGray)
    return cl1


if __name__ == '__main__':
    img = cv2.imread('einstein-low-contrast.tif')
    retSplit = split(img[:])
    lib = CallLibrary(img[:])
    cli = CLAHE(img[:])
    cv2.imwrite('split.png', retSplit)
    cv2.imwrite('lib.png', lib)
    cv2.imwrite('cli.png', cli)






