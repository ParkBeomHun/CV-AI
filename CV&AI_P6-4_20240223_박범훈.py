import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound

class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('교통 약자 보호')
        self.setGeometry(200,200,700,200)

        signButton = QPushButton('표지판 등록',self)
        roadButton = QPushButton('도로 영상 불러옴',self)
        recognitionButton = QPushButton('인식',self)
        quitButton = QPushButton('나가기',self)
        self.label=QLabel('환영합니다!',self)

        signButton.setGeometry(10,10,100,30)
        roadButton.setGeometry(110,10,100,30)
        recognitionButton.setGeometry(210,10,100,30)
        quitButton.setGeometry(510,10,100,30)
        self.label.setGeometry(10,40,600,170)

        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles=[['child.png','어린이'],['elder.png','노인'],['disabled.png','장애인']]
        self.signImgs = []

    def signFunction(self): #입력 도로 이미지에서 찾을 표지판 등록 함수
        self.label.clear()
        self.label.setText('교통약자 표지판을 등록합니다.')

        for fname,_ in self.signFiles:
            self.signImgs.append(cv.imread(fname))  #어린이, 노인, 장애인 표지판을 signImgs 목록에 넣음
            cv.imshow(fname,self.signImgs[-1])      #각 표지판 이미지 출력

    def roadFunction(self): #입력할 도로 이미지 선택하는 함수
        if self.signImgs==[]:
            self.label.setText('먼저 표지판을 등록하세요.') #도로 이미지에서 찾을 표지판 이미지가 등록되어있지 않으면 표지판 먼저 등록하라고 말함
        else:
            fname=QFileDialog.getOpenFileName(self,'파일 읽기','./')    #도로 이미지로 사용할 이미지 선택
            self.roadImg=cv.imread(fname[0])
            if self.roadImg is None: sys.exit('파일을 찾을 수 없습니다.')

            cv.imshow('Road Scene',self.roadImg)    #선택한 도로 이미지 출력

    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText('먼저 도로 영상을 입력하세요.')
        else:
            sift=cv.SIFT_create()

            KD=[]   #표지판 이미지들에서 각 표지판의 특징점과 기술자를 추출해서 저장할 List 생성(Keypoints and Descriptor)
            for img in self.signImgs:
                gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)   #각 표지판 이미지를 GrayScale로 바꿔줌
                KD.append(sift.detectAndCompute(gray,None)) #각 표지판 이미지의 grayScale 버전에서 특징점과 기술자를 찾아서 List KD에 넣어줌 
            
            grayRoad = cv.cvtColor(self.roadImg,cv.COLOR_BGR2GRAY)  #입력한 도로 이미지를 GrayScale로 변환
            road_kp,road_des = sift.detectAndCompute(grayRoad,None) #도로 이미지의 GrayScale 버전에서 특징점과 기술자를 찾아서 저장

            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)  #FLANN 기반으로 매칭을 위한 matcher를 생성해줌
            GM=[]   #3장의 표지판 이미지와 선택한 도로 이미지를 매칭시킨 결과들을 저장할 List GM 설정(GoodMatches)
            for sign_kp, sign_des in KD:    #표지판 이미지들의 특징점과 기술자를 사용하기 위한 반복문
                knn_match=matcher.knnMatch(sign_des,road_des,2) #표지판 이미지의 기술자와 도로 이미지의 기술자에서 매칭되는 것을 찾음
                T=0.7
                good_match=[]   #매칭된 것들을 저장할 list good_match 선언
                for nearest1, nearest2 in knn_match:    
                    if (nearest1.distance/nearest2.distance)<T:
                        good_match.append(nearest1) #매칭된 점들을 good_match list에 추가
                GM.append(good_match)   #good_match를 GM에 넣음
            
            best = GM.index(max(GM,key=len))    #매칭된 쌍이 가장 많은 표지판의 index를 best로 선언 

            if len(GM[best])<4: #제일 매칭이 많이 된 표지판의 매칭 쌍이 4보다 작으면 매칭된 표지판이 없다고 말하는 조건문
                self.label.setText('표지판이 없습니다.')
            else:               #제일 매칭이 많이 된 표지판의 매칭 쌍이 4보다 크면 실행되는 조건문
                sign_kp=KD[best][0]     #각 표지판의 특징점들 중에 매칭이 제일 많이 된 표지판의 특징점을 sign_kp에 저장
                good_match=GM[best]     #가장 매칭이 잘된 표지판의 매칭점들을 good_match에 저장 

                points1=np.float32([sign_kp[gm.queryIdx].pt for gm in good_match])  #point1을 잘 매칭된 매칭쌍들 중 표지판쪽의 포인트로 선언
                points2=np.float32([road_kp[gm.queryIdx].pt for gm in good_match])  #point2을 잘 매칭된 매칭쌍들 중 도로쪽의 포인트로 선언

                H,_=cv.findHomography(points1, points2,cv.RANSAC)   #RANSAN 알고리즘으로 호모그래피 행렬을 추정해 H에 저장한다

                h1,w1=self.signImgs[best].shape[0], self.signImgs[best].shape[1]    #매칭된 표지판 이미지의 크기 추출
                h2,w2=self.roadImg.shape[0],self.roadImg.shape[1]                   #도로 이미지의 크기 추출

                box1=np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(4,1,2)   #표지판 이미지의 크기로 box1 만듬
                box2=cv.perspectiveTransform(box1,H)    #표지판 이미지에서 추출한 box1을 도로 이미지에 호모그래피 행렬을 통해 투영한 모양을 box2로 만듬

                self.roadImg=cv.polylines(self.roadImg,[np.int32(box2)],True,(0,255,0),4)   #도로 이미지에 투영된 표지판 모양대로 도로 이미지에 상에 그림

                img_match=np.empty((max(h1,h2),w1+w2,3),dtype=np.uint8) 
                cv.drawMatches(self.signImgs[best],sign_kp,self.roadImg,road_kp,good_match,img_match,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  
                cv.imshow('Matches and Homography',img_match)   #두 이미지의 매칭들을 표현하기위해 두 이미지를 합친 이미지와 매칭점들을 표시함

                self.label.setText(self.signFiles[best][1]+'보호구역입니다. 30km로 서행하세요.')    #어떤 표지판이 검출되었는지 띄우고 서행하라고 알림
                winsound.Beep(3000,500) #보호구역 표지판이 확인되었기 때문에 경고음으로 알림

    def quitFunction(self): #나가기 버튼 누르면 종료
        cv.destroyAllWindows()
        self.close()



app = QApplication(sys.argv)
win=TrafficWeak()
win.show()
app.exec_()