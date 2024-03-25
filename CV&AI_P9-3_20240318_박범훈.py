import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():
    f=open(r'C:\Users\IT\Desktop\CV&AI\sample\ch9\coco_names.txt', 'r')     #coco_names 데이터셋의 부류 이름파일 불러옴
    class_names=[line.strip() for line in f.readlines()]                    #coco_names의 이름들을 저장해줌

    model=cv.dnn.readNet(r'C:\Users\IT\Desktop\CV&AI\sample\ch9\yolov3.weights', r'C:\Users\IT\Desktop\CV&AI\sample\ch9\yolov3.cfg') #모델의 weight와 구조를 받아옴
    layer_names=model.getLayerNames()   
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model,out_layers,class_names #모델, 층, 부류 이름들을 반환해줌

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]  #입력 이미지의 크기 
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)    #YOLO의 입력에 맞게 이미지 변환
    
    yolo_model.setInput(test_img)   #변환된 이미지를 YOLO의 입력으로 넣음
    output3=yolo_model.forward(out_layers)  #YOLO의 출력을 구함(bounding box의 크기와 좌표, confidence score, class name)
    
    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3:      #output3에 들어있는 각 출력(여러가지 물체들이 있을 것임)에 대해서 반복함
        for vec85 in output:
            scores=vec85[5:]            #각 검출들의 confidence score들을 다 가져옴
            class_id=np.argmax(scores)  #각 검출들의 최대class 확률 값으로 class id 설정
            confidence=scores[class_id] #각 검출의 confidence score 빼줌
            if confidence>0.5:	        #신뢰도가 50% 이상인 경우만 취함
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)    #각 검출의 중심 좌표
                w,h=int(vec85[2]*width),int(vec85[3]*height)                #각 검출의 너비와 높이
                x,y=int(centerx-w/2),int(centery-h/2)                       #각 검출의 좌측 상단 좌표
                box.append([x,y,x+w,y+h])                                   #각 검출의 box 정보 넣어줌
                conf.append(float(confidence))                              #각 검출의 confidence score 넣어줌
                id.append(class_id)                                         #각 검출의 class_id 넣어줌
            
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4)   #NMS 알고리즘을 통해 class 신뢰도 임계값 0.4 / IoU 임계값 0.5로 설정해서 후보군을 추림
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind] #위의 줄로 최종 결과물(최종 bounding box들)로 object 변수 만들어줌
    return objects

model,out_layers,class_names=construct_yolo_v3()		# YOLO 모델 생성
colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔 random 하게 부여

cap=cv.VideoCapture(0,cv.CAP_DSHOW)
if not cap.isOpened(): sys.exit('카메라 연결 실패')

import time

start=time.time()
n_frame=0

while True:
    ret,frame=cap.read()
    if not ret: sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')
        
    res=yolo_detect(frame,model,out_layers)   
 
    for i in range(len(res)):
        x1,y1,x2,y2,confidence,id=res[i]
        text=str(class_names[id])+'%.3f'%confidence
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[id],2)
        cv.putText(frame,text,(x1,y1+30),cv.FONT_HERSHEY_PLAIN,1.5,colors[id],2)
    
    cv.imshow("Object detection from video by YOLO v.3",frame)
    n_frame+=1
    
    key=cv.waitKey(1) 
    if key==ord('q'): break 

end=time.time()
print('처리한 프레임 수 = ',n_frame,', 경과시간 = ',end-start,'\n초당 프레임 수 =',n_frame/(end-start))

cap.release()		# 카메라와 연결을 끊음
cv.destroyAllWindows()