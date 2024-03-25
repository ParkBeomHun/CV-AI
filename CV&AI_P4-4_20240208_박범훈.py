import cv2 as cv
import sys

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\apples.jpg')
if img is None:
    print('이미지를 읽을수 없습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
apples = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,200,param1=150,param2=20,minRadius=50,maxRadius=120)
#param1 전에꺼 : 원 사이의 최소 거리
#param1 : canny edge 알고리즘의 임계값
#param2 : Hough Transform에서 사용하는 비최대 억제를 적용할때 쓰는 입계값

for i in apples[0]:
    cv.circle(img,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)

cv.imshow('Apples detection',img)

cv.waitKey()
cv.destroyAllWindows()