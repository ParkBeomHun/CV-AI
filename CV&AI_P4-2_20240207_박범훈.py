import cv2 as cv
import sys

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')
if img is None:
    print('이미지를 읽을수 없습니다.')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

canny1 = cv.Canny(gray,50,150)      #T(high) : 최초 기준 임계값(엣지 강도가 150 이상인 부분을 엣지로 가정하고 해당 부분에서 엣지 추적 시작함)
canny2 = cv.Canny(gray,100,200)     #T(Low) : 최초 기준 임계값 근처라면 T(Low)만 넘으면 엣지로 판단

cv.imshow('Original img',gray)
cv.imshow('Canny 1 img',canny1)
cv.imshow('Canny 2 img',canny2)

cv.waitKey()
cv.destroyAllWindows()