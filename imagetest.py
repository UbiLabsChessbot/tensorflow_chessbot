#coding:utf-8
#import
import cv2
import numpy as np

img = cv2.imread("q5.png")
res=cv2.resize(img,(94,94),interpolation=cv2.INTER_CUBIC)
hsv = cv2.cvtColor(res,cv2.COLOR_BGR2HSV)
lower_red = np.array([156,80,80])
upper_red = np.array([200,255,255])
mask = cv2.inRange(hsv,lower_red,upper_red)
kernel = np.ones((2,2),np.uint8)
erosion = cv2.erode(mask,kernel,iterations=1)

gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

cnt = contours[0]
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
print center
print img[int(x),int(y)]
# gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.namedWindow("image")
cv2.imshow("image",erosion)
cv2.waitKey(0)

# cnt = contours[0]
# (x,y),radius = cv2.minEnclosingCircle(cnt)
# center = (int(x),int(y))
# Ang = np.arctan((47-x)/(47-y))
# degree=np.abs((Ang*180)/np.pi)
# if Ang>0:
#   degree=360-degree
# M= cv2.getRotationMatrix2D((47,47),degree,1)
# dst = cv2.warpAffine(res,M,(94,94))
# for i in range(94):
#     for j in range(94):
#         if dst[i,j,0]==0:
#             dst[i,j]=[181,217,240]
# img2[6:100, 6:100 ] = dst
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow("Image")
# cv2.imshow("Image",dst)
# cv2.namedWindow("Image2")
# cv2.imshow("Image2",img2)
# cv2.imwrite('q12.png',img2)
# cv2.waitKey(0)
