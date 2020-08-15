import cv2
import numpy as np

img_rgb = cv2.imread('allWaldo.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('waldo-test.PNG',0)
w, h = template.shape[::-1]

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
# https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.4
loc = np.where( res >= threshold)

#Zip comvierte el arreglo recibido en tupletas (w,h)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)


cv2.imshow('Detected',img_rgb)
cv2.imshow('test',template)

cv2.waitKey(0)
cv2.destroyAllWindows()

