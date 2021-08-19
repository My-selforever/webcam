from os import access
import cv2
import time
import numpy as np
oncam = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
opfile = cv2.VideoWriter('testvid.avi',fourcc,20.0,(640,480))

#rest
time.sleep(2)

#capture masking image
maskn = 0

for i in range(60):
    read,maskn = oncam.read()

#flip

maskn = np.flip(maskn,axis=1)

while(oncam.isOpened()):
    read,vid = oncam.read()
    if(read==False):
        break
    #flip
    vid = np.flip(vid,axis=1)

    #convert to Hue Saturation Value format
    hsv = cv2.cvtColor(vid,cv2.COLOR_BGR2HSV)
    #color array
    lr = np.array([0,120,50])
    ur = np.array([10,255,255])
    #masking 
    mask1 = cv2.inRange(hsv,lr,ur)

    #masking 2
    lr = np.array([170,120,50])
    ur = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lr,ur)

    mask1 = mask1+mask2

    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    
    #removing uncommon mask1 and storing on mask2
    mask2 = cv2.bitwise_not(mask1)

    #result without the given color
    result1 = cv2.bitwise_and(vid,vid,mask=mask2)
    result2 = cv2.bitwise_and(vid,vid,mask=mask1)
    fo = cv2.addWeighted(result1,1,result2,1,0)

    opfile.write(fo)

    cv2.imshow("Invisiblity Cape",fo)
    cv2.waitKey(1)

oncam.release()
cv2.destroyAllWindows()
opfile.release()