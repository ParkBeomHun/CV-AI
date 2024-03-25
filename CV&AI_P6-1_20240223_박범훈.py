from PyQt5.QtWidgets import *
import sys
import winsound

class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('삑 소리 내기')      #제목이 '삑 소리 내기'인 Window창 만들기
        self.setGeometry(200,200,500,100)       #Window 창 크기를 (100,500)으로 만들고 초기 위치를 (200,200)으로 설정

        shortBeepButton = QPushButton('짧게 삑', self)  #짧게 삑 이라는 버튼 추가
        longBeepButton = QPushButton('길게 삑', self)   #길게 삑 이라는 버튼 추가
        quitButton=QPushButton('나가기', self)          #나가기 버튼 추가
        self.label=QLabel('환영합니다!',self)           #환영합니다 라는 문구 추가

        shortBeepButton.setGeometry(10,10,100,30)      #짧게 삑 이라는 버튼 크기와 위치 설정
        longBeepButton.setGeometry(110,10,100,30)      #길게 삑 이라는 버튼 크기와 위치 설정
        quitButton.setGeometry(210,10,100,30)          #나가기 버튼 크기와 위치 설정
        self.label.setGeometry(10,40,500,70)           #환영합니다 라는 문구 크기와 위치 설정

        shortBeepButton.clicked.connect(self.shortBeepFunction) #짧게 삑이라는 버튼 눌렸을때 함수와 연결
        longBeepButton.clicked.connect(self.longBeepFunction)   #길게 삑이라는 버튼 눌렸을때 함수와 연결
        quitButton.clicked.connect(self.quitFunction)           #나가기 버튼 눌렸을때 함수와 연결

    def shortBeepFunction(self):    #짧게 삑 버튼 눌렀을시 실행되는 함수 설정
        self.label.setText('주파수 1000으로 0.5초 동안 삑 소리를 냅니다.')
        winsound.Beep(1000,500)
        
    def longBeepFunction(self):     #길게 삑 버튼 눌렀을시 실행되는 함수 설정
        self.label.setText('주파수 1000으로 3초 동안 삑 소리를 냅니다.')
        winsound.Beep(1000,3000)

    def quitFunction(self):         #나가기 버튼 눌렀을시 실행되는 함수 설정
        self.close()

app = QApplication(sys.argv)    #PyQt실행에 필요한 객체 생성
win = BeepSound()               #BeepSoubnd 전체 class 실행
win.show()                      #win에 해당하는 윈도우를 실제로 출력시킴
app.exec()                      #무한 루프를 돌아 프로그램이 끝나는것 방지