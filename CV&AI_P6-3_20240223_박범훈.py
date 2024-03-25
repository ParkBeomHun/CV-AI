import cv2 as cv
import numpy as np
import sys
from PyQt5.QtWidgets import *

class Orim(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('오림')
        self.setGeometry(200,200,700,200)

        fileButton = QPushButton('파일',self)
        paintButton = QPushButton('페인팅',self)
        cutButton = QPushButton('오림',self)
        incButton = QPushButton('+',self)
        decButton = QPushButton('-',self)
        saveButton = QPushButton('저장',self)
        quitButton = QPushButton('나가기',self)

        fileButton.setGeometry(10,10,100,30)
        paintButton.setGeometry(110,10,100,30)
        cutButton.setGeometry(210,10,100,30)
        incButton.setGeometry(310,10,50,30)
        decButton.setGeometry(360,10,50,30)
        saveButton.setGeometry(410,10,100,30)
        quitButton.setGeometry(510,10,100,30)

        fileButton.clicked.connect(self.fileOpenFunction)   #Orim에 사용할 이미지 파일 불러오기
        paintButton.clicked.connect(self.paintFunction)     #Orim을 위한 페인트 칠한다고 선언하는 버튼
        cutButton.clicked.connect(self.cutFunction)         #Painting을 기반으로 Orim 진행
        incButton.clicked.connect(self.incFunction)         #Painting을 위한 브러쉬 크기 키우기
        decButton.clicked.connect(self.decFunction)         #Painting을 위한 브러쉬 크기 줄이기
        saveButton.clicked.connect(self.saveFunction)       #Orim한 이미지 저장하기
        quitButton.clicked.connect(self.quitFunction)       #Orim 프로그램 종료

        self.BrushSiz = 5
        self.LColor,self.RColor = (255,0,0),(0,0,255)

    def fileOpenFunction(self):
        fname=QFileDialog.getOpenFileName(self,'Open File','./')    #파일 선택
        self.img = cv.imread(fname[0])                              #선택한 파일 이름 읽음
        if self.img is None: sys.exit('파일을 찾을 수 없습니다.')

        self.img_show=np.copy(self.img)         
        cv.imshow('Painting',self.img_show)     #선택한 파일 카피해서 띄움

        self.mask = np.zeros((self.img.shape[0],self.img.shape[1]),np.uint8)    #빨간색 파란색으로 painting 했을때 색칠 정보를 담을 객체 생성
        self.mask[:,:] = cv.GC_PR_BGD

    def paintFunction(self):
        cv.setMouseCallback('Painting',self.painting)
    
    def painting(self,event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.LColor,-1)
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_FGD,-1)
        elif event == cv.EVENT_RBUTTONDOWN:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.RColor,-1)
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_BGD,-1)               #우클릭시 빨간색으로 배경 painting
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.LColor,-1)
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_FGD,-1)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.RColor,-1)
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_BGD,-1)               #좌클릭시 파랑색으로 객체 painting

        cv.imshow('Painting',self.img_show)                                     #painting 한 그림 보여줌
    
    def cutFunction(self):
        background = np.zeros((1,65),np.float64)
        foreground = np.zeros((1,65),np.float64)
        cv.grabCut(self.img, self.mask, None, background,foreground,5,cv.GC_INIT_WITH_MASK) #빨간색 파란색으로 painting한 객체와 배경으로 grabcut 실행
        mask2 = np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8')
        self.grabImg = self.img*mask2[:,:,np.newaxis]   #배경으로 선정된 부분을 mask2와 곱해서 검은색으로 처리
        cv.imshow('Scissoring',self.grabImg)            #GrabCut된 이미지를 출력

    def incFunction(self):
        self.BrushSiz = min(20,self.BrushSiz+1) #BrushSize 키우기

    def decFunction(self):
        self.BrushSiz = max(1,self.BrushSiz-1)  #BrushSize 줄이기
        
    def saveFunction(self):
        fname=QFileDialog.getSaveFileName(self,'파일 저장','./')    #GrabCut한 이미지 저장
        cv.imwrite(fname[0],self.grabImg)

    def quitFunction(self):     #Orim 프로그램 나가기
        cv.destroyAllWindows()
        self.close()

app=QApplication(sys.argv)
win = Orim()
win.show()
app.exec_()