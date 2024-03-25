from pixellib.semantic import semantic_segmentation
import cv2 as cv

cap = cv.VideoCapture(0)    #웹캠과 연결해서 결과를 cap에 저장

seg_video=semantic_segmentation()                                   #semantic_segmentation 객체 seg 생성
seg_video.load_ade20k_model('deeplabv3_sception65_ade20k.h5')       #학습이 되어 있는 모델을 불러옴

seg_video.process_camera_ade20k(cap, overlay=True,frames_per_second=2,output_video_name='output_video.mp4',show_frames=True,frame_name='Pixellib')
#영상 정보를 분할시키고 초당 2프레임으로 영상을 저장시킴
cap.release()
cv.destroyAllwindows()