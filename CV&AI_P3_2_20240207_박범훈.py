import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg')
h = cv.calcHist([img],[2],None,[256],[0,256])   #img의 2번째 채널(R) 0~255로 표현된 각 픽셀들을 256개의 칸으로 나눠서 각각 얼만큼의 빈도로 나오는지 확인
plt.plot(h,color = 'r',linewidth=1)
plt.show()