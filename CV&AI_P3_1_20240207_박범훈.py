import cv2 as cv
import sys

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')
if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

#print(img.shape) #(372, 559, 3) : (y,x,C)

cv.imshow('OG_RGB',img)
cv.imshow('Upper Right half of img',img[    :   ,       img.shape[1]//2   :     -1        ,         :     ])    # (0,0)에서 이미지의 높이 절반, 너비 절반까지 보여주는 부분 (+a -1은 끝까지)
"""
cv.imshow('Center half of img',img[img.shape[0]//4:3*img.shape[0]//4, img.shape[1]//4:3*img.shape[1]//4])   # 이미지의 x축 y축 상의 1/4 ~ 3/4만큼 출력시켜서 이미지의 정중앙 부분을 출력

cv.imshow('R channel',img[:,:,2])   # 전체 이미지를 보여줘야해서 x축 y축은 :로 설정 / R채널을 보여줘야해서 BGR중 R-Channel 출력
cv.imshow('G channel',img[:,:,1])   # 전체 이미지를 보여줘야해서 x축 y축은 :로 설정 / G채널을 보여줘야해서 BGR중 G-Channel 출력
cv.imshow('B channel',img[:,:,0])   # 전체 이미지를 보여줘야해서 x축 y축은 :로 설정 / B채널을 보여줘야해서 BGR중 B-Channel 출력
"""
cv.waitKey()
cv.destroyAllWindows()