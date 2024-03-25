import cv2 as cv
import numpy as np
import sys

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')
if img is None:
    print('이미지를 읽을수 없습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray,100,200)

contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#contour = 윤곽선들의 집합
#contour[i] = 특정 윤곽선
#hierarchy = 윤곽선들의 계층적 구조를 설명하는 배열이라는데 뭔지 모르겠음


lcontour = []
for i in range(len(contour)):
    if contour[i].shape[0] > 100:
        lcontour.append(contour[i])
#같은 외곽선으로 분류되는 것들중 100개 이상의 점으로 구성되는 외곽선만 lcontour에 추가해줌


cv.drawContours(img,lcontour,-1,(0,255),3)

cv.imshow('Original with contours',img)
cv.imshow('Canny',canny)

cv.waitKey()
cv.destroyAllWindows()