import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
from pixellib.tune_bg import alter_bg
    
class VideoSpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('배경을 내 맘대로')     #배경을 내 맘대로 라는 창 만듬
        self.setGeometry(200,200,400,100)          #크기 설정   
       
        videoButton=QPushButton('배경 내 맘대로 켜기',self)
        self.pickCombo=QComboBox(self)     
        self.pickCombo.addItems(['원래 영상','흐릿(조금)','흐릿(중간)','흐릿(많이)','빨강','녹색','파랑'])
        quitButton=QPushButton('나가기',self)        
        
        videoButton.setGeometry(10,10,140,30)
        self.pickCombo.setGeometry(150,10,110,30)                  
        quitButton.setGeometry(280,10,100,30)
        
        videoButton.clicked.connect(self.videoSpecialEffectFunction) 
        quitButton.clicked.connect(self.quitFunction)
        
    def videoSpecialEffectFunction(self):             
        self.cap=cv.VideoCapture(0,cv.CAP_DSHOW)                    #웹캠과 연결
        if not self.cap.isOpened(): sys.exit('카메라 연결 실패')
        
        while True:
            ret,frame=self.cap.read()  #웹캠에서 프레임 읽어옴
            if not ret: break

            pick_effect=self.pickCombo.currentIndex()        
            if pick_effect==0:
                special_img=frame                                                               #원래 이미지 그대로 출력
            elif pick_effect==1:
                special_img=change_bg.blur_frame(frame,low=True,detect='person')                #약간 흐릿하게
            elif pick_effect==2:
                special_img=change_bg.blur_frame(frame,moderate=True,detect='person')           #중간 흐릿하게 
            elif pick_effect==3:    
                special_img=change_bg.blur_frame(frame,extreme=True,detect='person')            #완전 흐릿하게
            elif pick_effect==4:    
                special_img=change_bg.color_frame(frame,colors=(255,0,0),detect='person')       #배경을 빨간색으로
            elif pick_effect==5:    
                special_img=change_bg.color_frame(frame,colors=(0,255,0),detect='person')       #배경을 초록색으로
            elif pick_effect==6:    
                special_img=change_bg.color_frame(frame,colors=(0,0,255),detect='person')       #배경을 파란색으로
                
            cv.imshow('Special effect',special_img)                                             #설정한 대로 이미지 출력 
            cv.waitKey(1) 
                                        
    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()        
        self.close()

change_bg=alter_bg(model_type="pb")
change_bg.load_pascalvoc_model('xception_pascalvoc.pb')     #분할할때 사용한 모델 불러옴
                
app=QApplication(sys.argv) 
win=VideoSpecialEffect() 
win.show()
app.exec_()