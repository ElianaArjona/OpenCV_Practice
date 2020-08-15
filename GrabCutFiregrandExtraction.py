import numpy as np
import cv2
import matplotlib.pyplot as plt

#img = cv2.imread('opencv-python-foreground-extraction-tutorial.jpg')
img = cv2.imread('rodolfin.png')
mask= np.zeros(img.shape[:2],np.uint8)

print(mask)

bgModel = np.zeros((1,65),np.float64)
fgModel = np.zeros((1,65),np.float64)

print(fgModel)

# rect = (start_x, start_y, width, height).
rect = (170,20,230,200)

cv2.grabCut(img,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()