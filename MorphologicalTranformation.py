import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red = np.array([60,30,30])
    upper_red = np.array([180,255,250])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations=1)
    dilation = cv2.dilate(mask,kernel,iterations=1)

    #remove false psositive
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

    #remove false negative
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    cv2.imshow('Original', frame)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
    # cv2.imshow('dilation', dilation)
    # cv2.imshow('kernel', kernel)
    # cv2.imshow('res', res)
    # cv2.imshow('erosion', erosion)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()