from pixellib.semantic import semantic_segmentation
import cv2 as cv

seg=semantic_segmentation() #semantic_segmentation 객체 seg 생성
seg.load_ade20k_model('deeplabv3_xception65_ade20k.h5') #학습이 되어 있는 모델을 불러옴

img_fname='busy_street.jpg'
seg.segmentAsAde20k(img_fname,output_image_name='image_new.jpg')        #원본 이미지를 분할 시킨 이미지를 image_new.jpg로 저장
info1,img_segmented1=seg.segmentAsAde20k(img_fname)                     #원본 이미지를 분할시킴
info2,img_segmented2=seg.segmentAsAde20k(img_fname,overlay=True)        #분할하는데 원본 영상을 투영시켜서 원본 이미지를 보이게함

cv.imshow('Image original',cv.imread(img_fname))            #원본 이미지 출력
cv.imshow('Image segmention',img_segmented1)                #분할된 이미지 출력
cv.imshow('Image segmention overlayed',img_segmented2)      #분할된 이미지에 원본 이미지 투영해서 출력

cv.waitKey()
cv.destroyAllWindows()