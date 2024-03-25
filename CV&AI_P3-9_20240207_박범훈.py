import cv2 as cv
import numpy as np
import time

def my_cvtGray1(bgr_img):
    g = np.zeros([bgr_img.shape[0],bgr_img.shape[1]])
    for r in range(bgr_img.shape[0]):
        for c in range(bgr_img.shape[1]):
            g[r,c] = 0.144*bgr_img[r,c,0]+0.587*bgr_img[r,c,1]+0.299*bgr_img[r,c,2]

    return np.uint8(g)

def my_cvtGray2(bgr_img):
    g = np.zeros([bgr_img.shape[0],bgr_img.shape[1]])
    g = 0.144*bgr_img[:,:,0]+0.587*bgr_img[:,:,1]+0.299*bgr_img[:,:,2]
    return np.uint8(g)

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')

start = time.time()
my_cvtGray1(img)
print('My time1 : ',time.time()-start)      #내가 만든 이미지로 for문 돌림

start = time.time()
my_cvtGray2(img)
print('My time2 : ',time.time()-start)      #파이썬 기능으로 처리

start = time.time()
cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print('OpenCV time : ',time.time()-start)   #openCV 사용

