import numpy as np
import cv2

cap = cv2.VideoCapture('people-walking.mp4')
#cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while (1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    #res = cv2.bitwise_and(frame, frame, mask=fgmask)
    #median = cv2.medianBlur(res, 3)

    cv2.imshow('orginal', frame)

    #cv2.imshow('MedianBlurMovement', median)
    cv2.imshow('frameMask', fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()