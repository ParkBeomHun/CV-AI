import cv2 as cv
import numpy as np

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg',cv.IMREAD_UNCHANGED)
img = cv.resize(img,dsize=(0,0),fx=0.5,fy=0.5)

def gamma(f,gamma=1.0):
    f1 = f/255.0
    return np.uint8(255*(f1**gamma))

gc = np.hstack((gamma(img,0.5),gamma(img,0.75),gamma(img,1.0),gamma(img,2.0),gamma(img,3.0)))
cv.imshow('gamma',gc)

cv.waitKey()
cv.destroyAllWindows()