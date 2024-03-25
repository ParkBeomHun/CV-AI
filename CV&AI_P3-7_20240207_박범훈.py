import cv2 as cv
import numpy as np

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg',cv.IMREAD_UNCHANGED)
img = cv.resize(img,dsize=(0,0),fx = 0.5 , fy = 0.5)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.putText(gray, 'drogba', (10,20),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
cv.imshow('Original img',gray)

smooth = np.hstack((cv.GaussianBlur(gray,(5,5),0.0), cv.GaussianBlur(gray,(9,9),0.0), cv.GaussianBlur(gray,(15,15),0.0)))
cv.imshow('Smoothen img',smooth)

emboss_filter = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])

gray16 = np.int16(gray)
emboss = np.uint8(np.clip(cv.filter2D(gray16,-1,emboss_filter)+128,0,255))  #원래의 엠보싱 필터링 방법대로 한 것
emboss_bad = np.uint8(cv.filter2D(gray16,-1,emboss_filter)+128)             #0~255범위 이외 범위를 0 or 255로 표현하는 클리핑을 하지 않은 엠보싱 방법
emboss_worse = cv.filter2D(gray,-1,emboss_filter)                           #애초에 음수로 표현될수 없는 자료형에다가 엠보싱 필터를 적용시켰음

cv.imshow('Emboss',emboss)
cv.imshow('Emboss Bad',emboss_bad)
cv.imshow('Emboss Worse',emboss_worse)

cv.waitKey()
cv.destroyAllWindows()