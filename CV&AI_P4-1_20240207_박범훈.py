import cv2 as cv
import sys

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')
if img is None:
    print('이미지를 읽을수 없습니다.')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray,cv.CV_32F,1,0,ksize=3)
grad_y = cv.Sobel(gray,cv.CV_32F,0,1,ksize=3)

sobel_x = cv.convertScaleAbs(grad_x)
sobel_y = cv.convertScaleAbs(grad_y)

edge_strength = cv.addWeighted(sobel_x,0.5, sobel_y,0.5,0)

cv.imshow('Original img',gray)
cv.imshow('Sobel Gradient for x',sobel_x)
cv.imshow('Sobel Gradient for y',sobel_y)
cv.imshow('Edge Strength',edge_strength)

cv.waitKey()
cv.destroyAllWindows()
