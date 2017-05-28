import cv2
import numpy as np
dst = cv2.imread("Chess.png")
color = [181,217,240]
#let image get darker
for m in range(360):
    for n in range(720):
        dst[m,n,0]=int(dst[m,n,0]*0.9)
        dst[m,n,1]=int(dst[m,n,1]*0.9)
        dst[m,n,2]=int(dst[m,n,2]*0.9)
    #show the process
cv2.namedWindow('img')
cv2.imshow('img',dst)
cv2.waitKey()
# cv2.destroyAllWindows()
# print'.'
# print 'processing'
# for xi in xrange(0,w):
#     for xj in xrange(0,h):
#         ##set the pixel value increase to 1020%
#         img[xj,xi,0] = int(img[xj,xi,0]*10.2)
#         img[xj,xi,1] = int(img[xj,xi,1]*10.2)
#         img[xj,xi,2] = int(img[xj,xi,2]*10.2)
#         #show the process
#     if xi%10==0 :print '.',
# cv2.namedWindow('img')
# cv2.imshow('img',img)
# cv2.waitKey()
# cv2.destroyAllWindows()
