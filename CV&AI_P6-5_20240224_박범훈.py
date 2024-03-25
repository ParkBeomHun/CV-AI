from PyQt5.QtWidgets import *
import cv2 as cv
import numpy as np
import winsound
import sys
   
class Panorama(QMainWindow) :
    def __init__(self) :
        super().__init__()
        self.setWindowTitle('파노라마 영상')
        self.setGeometry(200,200,700,200)
        
        collectButton=QPushButton('영상 수집',self)
        self.showButton=QPushButton('영상 보기',self) 
        self.stitchButton=QPushButton('봉합',self) 
        self.saveButton=QPushButton('저장',self)
        quitButton=QPushButton('나가기',self)
        self.label=QLabel('환영합니다!',self)
        
        collectButton.setGeometry(10,25,100,30)
        self.showButton.setGeometry(110,25,100,30) 
        self.stitchButton.setGeometry(210,25,100,30) 
        self.saveButton.setGeometry(310,25,100,30)
        quitButton.setGeometry(450,25,100,30) 
        self.label.setGeometry(10,70,600,170)

        self.showButton.setEnabled(False)       #초기에 파노라마로 만들 이미지를 수집하지 않아서 선택하지 못하게 설정
        self.stitchButton.setEnabled(False) 
        self.saveButton.setEnabled(False)
        
        collectButton.clicked.connect(self.collectFunction)
        self.showButton.clicked.connect(self.showFunction)       
        self.stitchButton.clicked.connect(self.stitchFunction) 
        self.saveButton.clicked.connect(self.saveFunction)   
        quitButton.clicked.connect(self.quitFunction)        

    def collectFunction(self):                  #파노라마 이미지로 만들 여러 사진을 수집하는 함수
        self.showButton.setEnabled(False) 
        self.stitchButton.setEnabled(False) 
        self.saveButton.setEnabled(False)
        self.label.setText('c를 여러 번 눌러 수집하고 끝나면 q를 눌러 비디오를 끕니다.')    #파노라마 이미지로 만들 여러 이미지를 수집하는 방법 안내
        
        self.cap=cv.VideoCapture(0,cv.CAP_DSHOW) 
        if not self.cap.isOpened(): sys.exit('카메라 연결 실패')
        
        self.imgs=[]        #파노라마 이미지를 위해 필요한 이미지를 담을 list 생성
        while True:
            ret,frame=self.cap.read()  
            if not ret: break
            
            cv.imshow('video display', frame)
            
            key=cv.waitKey(1) 
            if key==ord('c'):            
                self.imgs.append(frame)	# 영상 저장(c를 누를시 촬영한 이미지들을 list에 저장)
            elif key==ord('q'):
                self.cap.release() 
                cv.destroyWindow('video display')                
                break 
        
        if len(self.imgs)>=2:		# 수집한 영상이 2장 이상이면 파노라마 영상을 만들수 있음
            self.showButton.setEnabled(True)        #촬영한 사진들 볼수있도록 하는 함수 활성화
            self.stitchButton.setEnabled(True)      #촬영한 사진들 함쳐주도록 하는 함수 활성화
            self.saveButton.setEnabled(True)        #합쳐진 사진들 
                    
    def showFunction(self):         # 수집한 영상 보여주는 함수
        self.label.setText('수집된 영상은 '+str(len(self.imgs))+'장 입니다.')   #수집한 영상의 총 갯수 출력
        stack=cv.resize(self.imgs[0],dsize=(0,0),fx=0.25,fy=0.25)              #수집한 영상들을 보여주기위햇 사이즈 조절
        for i in range(1,len(self.imgs)):
            stack=np.hstack((stack,cv.resize(self.imgs[i],dsize=(0,0),fx=0.25,fy=0.25))) #ressize된 수집된 영상들을 하나로 붙임
        cv.imshow('Image collection',stack)        
        
    def stitchFunction(self):
        stitcher=cv.Stitcher_create()   #파노라마 이미지 제작을 위한 영상 봉합 객체 stitcher 생성
        status,self.img_stitched=stitcher.stitch(self.imgs) #stitch 함수를 통해 수집된 이미지들 봉합
        if status==cv.STITCHER_OK:      #파노라마 이미지 제작을 성공한 경우
            cv.imshow('Image stitched panorama',self.img_stitched)     #봉합된 이미지를 출력함
        else:
            winsound.Beep(3000,500)            
            self.label.setText('파노라마 제작에 실패했습니다. 다시 시도하세요.')    
            
    def saveFunction(self):     #저장시 파노라마 이미지로 변환한 이미지 저장
        fname=QFileDialog.getSaveFileName(self,'파일 저장','./')
        cv.imwrite(fname[0],self.img_stitched)
        
    def quitFunction(self): 
        self.cap.release() 
        cv.destroyAllWindows()  
        self.close()

app=QApplication(sys.argv) 
win=Panorama() 
win.show()
app.exec_()