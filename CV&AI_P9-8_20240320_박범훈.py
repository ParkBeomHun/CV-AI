from pixellib.instance import instance_segmentation
import cv2 as cv

cap = cv.VideoCapture(0)    #웹캠과 연결

seg_video = instance_segmentation() #객체 분할할 객체 생성
seg_video.load_model('mask_rcnn_coco.h5')   #모델 불러옴

target_class=seg_video.select_target_classes(person=True,book=True)     #객채 분할할 class를 사람과 책으로 설정
seg_video.process_camera(cap,segment_target_classes=target_class, frames_per_second=2,show_frames=True,frame_name='Pixellib',show_bboxes=True)
#웹캠에서 읽어온것과 불러온 모델과 설정한 class들을 초당 2프레임으로 분할시킴
cap.release()
cv.destroyAllWindows()