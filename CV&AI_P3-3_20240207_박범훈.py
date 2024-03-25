import cv2 as cv
import sys


img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')

t,bin_img = cv.threshold(img[:,:,2],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
print('오츄 알고리즘이 찾은 최적의 임계값은 : ', t)

cv.imshow('R-Channel',img[:,:,2])
cv.imshow('R-Channel binarization',bin_img)

cv.waitKey()
cv.destroyAllWindows()