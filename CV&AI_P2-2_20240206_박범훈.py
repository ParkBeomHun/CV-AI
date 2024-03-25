import cv2 as cv
import sys

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')
if img is None:
    print("이미지를 읽을수 없습니다.")

cv.imshow('drogba img',img)

cv.waitKey()
cv.destroyAllWindows()