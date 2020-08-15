import cv2
import numpy as np

img1 = cv2.imread('bob_2_255.png')
#img2 = cv2.imread('bob_2_255.png')
img2 = cv2.imread('bob.png')#borra esto

#add = img1+img2
#add = cv2.add(img1,img2)

#weighted = cv2.addWeighted(img1,0.6,img2,0.4,0)
#cv2.imshow('add',weighted)

rows,cols,channels = img2.shape
roi = img1[0:rows,0:cols]

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)

#cv2.imshow('mask',mask)

mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
img2_fg = cv2.bitwise_and(img2,img2,mask=mask)

dst = cv2.add(img1_bg,img2_fg)
img1[0:rows,0:cols] = dst

cv2.imshow('rest',mask_inv)
cv2.imshow('mask',mask)
cv2.imshow('2img2fg',img2_fg)
cv2.imshow('1img1bg',img1_bg)
cv2.imshow('dst',dst)

cv2.waitKey(0)
cv2.destroyAllWindows()