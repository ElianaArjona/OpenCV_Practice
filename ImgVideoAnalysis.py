import cv2
import matplotlib.pyplot as plt
import urllib.request
import numpy as np

img = cv2.imread('bob.png',cv2.IMREAD_COLOR)
#to change color of pixel
#img[55,55] = [255,255,255]
#not change
px = img[55,55]

#pixel of a region
#img[5:150,20:150] = [255,255,0]
roi = img[5:150,20:150]
print(roi)

sponge_face = img[5:150,20:150]
img[0:145,0:130] = sponge_face

#plt.imshow(img,cmap='gray',interpolation='bicubic')
cv2.imshow('bob',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
