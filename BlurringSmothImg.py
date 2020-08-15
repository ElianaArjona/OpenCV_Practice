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

    kernel = np.ones((15, 15), np.float32) / 25
    smoothed = cv2.filter2D(res, -1, kernel)

    blur = cv2.GaussianBlur(res,(15,15),0)

    median = cv2.medianBlur(res,15)

    bilateral = cv2.bilateralFilter(res,15,75,75)

    cv2.imshow('Original', frame)
    cv2.imshow('smooth', smoothed)
    cv2.imshow('Blur', blur)
    cv2.imshow('median', median)
    cv2.imshow('bilateral', bilateral)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()