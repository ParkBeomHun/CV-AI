import cv2 as cv

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')
patch = img[250:350,170:270,:]

img = cv.rectangle(img,(170,250),(270,350),(255,0,0),3)
patch1 = cv.resize(patch,dsize=(0,0),fx = 5, fy = 5, interpolation = cv.INTER_NEAREST)  #최근접 이웃
patch2 = cv.resize(patch,dsize=(0,0),fx = 5, fy = 5, interpolation = cv.INTER_LINEAR)   #양선형 보간
patch3 = cv.resize(patch,dsize=(0,0),fx = 5, fy = 5, interpolation = cv.INTER_CUBIC)    #양3차 보간

cv.imshow('OG img',img)
cv.imshow('resize nearest',patch1)
cv.imshow('resize bilinear',patch2)
cv.imshow('resize bicubic',patch3)


cv.waitKey()
cv.destroyAllWindows()