import cv2
import numpy as np
img = cv2.imread("Chess.png")
img2 = cv2.imread("main.png")
color = [181,217,240]
for i in range(4):
    for j in range(8):
        test = img[i*90:(i+1)*90,j*90:(j+1)*90]
        res=cv2.resize(test,(100,100),interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(res,cv2.COLOR_BGR2HSV)
        lower_red = np.array([35,110,50])
        upper_red = np.array([99,255,255])
        mask = cv2.inRange(hsv,lower_red,upper_red)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        opened = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        closed = cv2.morphologyEx(opened,cv2.MORPH_OPEN,kernel)
        # res[closed==255]=[181,217,240]
        contours, hierarchy = cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        for t in range(len(contours)):
          if cv2.contourArea(cnt)<cv2.contourArea(contours[t]):
            cnt = contours[t]
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        cv2.circle(res,(50,50),50,(181,217,240),1)
        cv2.circle(res,center,int(radius),(0,0,0),-1)
        Ang = np.arctan((50-x)/(50-y))
        degree=np.abs((Ang*180)/np.pi)
        if Ang>0:
          degree=360-degree+90
        if x>50 and y>50:
          degree=degree-180
        if x<50 and y>50:
            degree=degree+180+90
        if x>50 and y<50:
            degree=degree+90
        M= cv2.getRotationMatrix2D((50,50),degree,1)
        dst = cv2.warpAffine(res,M,(100,100))

        for m in range(100):
            for n in range(100):
                if (dst[m,n,0]==0) and (dst[m,n,1]==0) and (dst[m,n,2]==0):
                    dst[m,n]=color
        if j%2==0:
           t=2*i
        else:
           t=2*i+1
        img2[j*100:(j+1)*100,t*100:(t+1)*100] = dst
GrayImage=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
retval, result = cv2.threshold(GrayImage, 147, 255, cv2.THRESH_BINARY)
cv2.namedWindow("Image2")
cv2.imshow("Image2",result)
cv2.waitKey()
