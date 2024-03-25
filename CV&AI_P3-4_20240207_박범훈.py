import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\IT\Desktop\CV&AI\img\drogba.jpg',cv.IMREAD_UNCHANGED)

t,bin_img = cv.threshold(img[:,:,2],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow("ORG",bin_img)
#plt.show()

se = np.uint8([
    [0,0,1,0,0],
    [0,1,1,1,0],
    [1,1,1,1,1],
    [0,1,1,1,0],
    [0,0,1,0,0]
])
"""
b_dilation = cv.dilate(bin_img,se,iterations=5)
plt.imshow(b_dilation,cmap='gray'),plt.xticks([]), plt.yticks([])#팽창
plt.show()

b_erosion = cv.erode(bin_img,se,iterations=1)
plt.imshow(b_erosion,cmap='gray'),plt.xticks([]), plt.yticks([])#수축
plt.show()
"""


b_closing = cv.erode(cv.dilate(bin_img,se,iterations = 5),se,iterations=5)#닫힘
cv.imshow("Closibng",b_closing)#,plt.xticks([]), plt.yticks([])
#plt.show()

cv.waitKey()
cv.destroyAllWindows()