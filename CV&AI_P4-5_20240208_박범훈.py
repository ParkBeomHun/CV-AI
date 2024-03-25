import skimage
import numpy as np
import cv2 as cv

img = skimage.data.coffee()
cv.imshow('Coffee image',cv.cvtColor(img,cv.COLOR_RGB2BGR))

slic1 = skimage.segmentation.slic(img,compactness = 20, n_segments = 600)
sp_img1 = skimage.segmentation.mark_boundaries(img,slic1)
sp_img1 = np.uint8(sp_img1*255.0)

slic2 = skimage.segmentation.slic(img,compactness = 40, n_segments = 600)
sp_img2 = skimage.segmentation.mark_boundaries(img,slic2)
sp_img2 = np.uint8(sp_img2*255.0)

cv.imshow('Super pixels (compact 20)',cv.cvtColor(sp_img1,cv.COLOR_RGB2BGR))
cv.imshow('Super pixels (compact 40)',cv.cvtColor(sp_img2,cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()

#slic알고리즘
#k개의 기준 픽셀들을 중심으로 주변의 픽셀들의 색, 거리 등을 통해서 가장 유사한 기준 픽셀들로 묶어주는것
#compactness : 슈퍼화소의 모양 조절, 값이 크면 슈퍼화소 사각형 가까워지지만 색의 유사성 떨어짐