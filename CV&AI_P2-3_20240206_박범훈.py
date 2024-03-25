import cv2 as cv
import sys

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')
if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)               #Color을 BGR에서 Gray scale로 바꿔줌
gray_small = cv.resize(gray,dsize=(0,0),fx=0.5, fy=0.5) #크기를 x축,y축 둘다 0.5 비율로 줄여주게 설정

cv.imwrite(r'C:\Users\IT\Desktop\CV&AI\img\drogba_gray.jpg',gray)
cv.imwrite(r'C:\Users\IT\Desktop\CV&AI\img\drogba_gray_small.jpg',gray_small)

cv.imshow('Color img',img)
cv.imshow('Gray img',gray)
cv.imshow('Gray and small img',gray_small)

cv.waitKey()
cv.destroyAllWindows()