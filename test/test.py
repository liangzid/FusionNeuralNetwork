import numpy as np
import cv2

'''
img1=cv2.imread('./1.png',cv2.IMREAD_COLOR)
img2=cv2.imread('./1.png',cv2.IMREAD_GRAYSCALE)
img3=cv2.imread('./1.png',cv2.IMREAD_UNCHANGED)
img4=cv2.imread('./1.png',0)
img5=cv2.imread('./1.png',1)

cv2.imshow('color',img1)
cv2.imshow('gray',img2)
cv2.imshow('unchanged',img3)
cv2.imshow('0',img5)
cv2.waitKey(0)

cv2.destroyAllWindows()
'''
cap=cv2.VideoCapture(0)

while(1):
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    print(ret)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break


cap.release()

cv2.destroyAllWindows()
