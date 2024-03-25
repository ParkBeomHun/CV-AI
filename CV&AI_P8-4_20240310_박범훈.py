import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import winsound
from keras.models import load_model

model = load_model('C:/Users/IT/Desktop/CV&AI/cnn_v2.h5')

def reset():
    global img

    img = np.ones((200,520,3),dtype=np.uint8)*255   #이미지를 다 흰색으로 만들기 위해서 255 곱해서 흰색 화면을 만들어줌
    for i in range(5):
        cv.rectangle(img,(10+i*100,50),(10+(i+1)*100,150),(0,0,255))    #5개의 숫자를 적을 사각형 만들어줌
    cv.putText(img,'e:erase s:show r:recognition q:quit',(10,40),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)   #각 버튼별 기능을 적어줌

def grab_numerals():
    numerals=[]
    for i in range(5):
        roi=img[51:149,11+i*100:9+(i+1)*100,0]  #각 사각형을 뗌
        roi=255-cv.resize(roi,(28,28),interpolation=cv.INTER_CUBIC) #28x28 크기로 변환을 해주고 처음에 검은색으로 그려져있어서 반전을 위해서 255에서 빼줘서 반전시켜준다
        numerals.append(roi)    #떼어낸 각각의 숫자들을 numerals에 추가해준다
    numerals=np.array(numerals) #붙어있는 숫자들을 numpy 배열로 반환한다
    return numerals

def show():
    numerals = grab_numerals()
    plt.figure(figsize=(25,5))  #다섯개의 숫자를 한번에 보여주기 위한 figure 생성
    for i in range(5):
        plt.subplot(1,5,i+1)    #각 figure들을 하나씩 추가해줌
        plt.imshow(numerals[i],cmap='gray')
        plt.xticks([]); plt.yticks([])  #x축 y축 눈금 제거 해줌
    plt.show()

def recognition():
    numerals=grab_numerals()    #각 숫자들을 받아옴
    numerals=numerals.reshape(5,28,28,1)    #각 숫자들을 28x28 -> 784 차원으로 바꿔줌
    numerals=numerals.astype(np.float32)/255.0  #각 숫자들을 정규화 해준다(0 or 1)
    res=model.predict(numerals)                 #해당 784차원으로 바꾼 특징을 가져온 모델에 집어넣어 어떤 숫자인지 예측한다.
    class_id=np.argmax(res,axis=1)              #res는 5x10차원(각 숫자들이 0~9중 어느것이 가장 유력한지 확률값들)이기 때문에 최고 높은 확률의 숫자로 예측해줌
    for i in range(5):
        cv.putText(img,str(class_id[i]),(50+i*100,180),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)   #각 숫자들의 예측값을 적어줌
    winsound.Beep(1000,500)

BrushSiz=4
LColor=(0,0,0)

def writing(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,LColor,-1)
    elif event == cv.EVENT_MOUSEMOVE and flags ==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img,(x,y),BrushSiz,LColor,-1)

reset()
cv.namedWindow('Writing')
cv.setMouseCallback('Writing',writing)

while(True):
    cv.imshow('Writing',img)
    key = cv.waitKey(1)
    if key == ord('e'):
        reset()
    elif key == ord('s'):
        show()
    elif key == ord('r'):
        recognition()
    elif key == ord('q'):
        break

cv.destroyAllWindows()