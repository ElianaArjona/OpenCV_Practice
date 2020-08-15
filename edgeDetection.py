import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.Canny(frame,100,50)

    cv2.imshow('Edges',edges)
    cv2.imshow('Original', frame)
    # cv2.imshow('laplacian', laplacian)
    # cv2.imshow('sobelx', sobelx)
    # cv2.imshow('sobely', sobely)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()