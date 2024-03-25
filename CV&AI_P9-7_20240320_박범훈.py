from pixellib.instance import instance_segmentation
import cv2 as cv

seg = instance_segmentation()
seg.load_model('mask_rcnn_coco.h5')

img_fname='busy_street.jpg' #이미지 불러옴
info, img_segmented = seg.segmentImage(img_fname, show_bboxes=True) #이미지를 객체 분할한 정보와 이미지를 받아옴

cv.imshow('Image segmention overlayed', img_segmented)

cv.waitKey()
cv.destroyAllWindows()
