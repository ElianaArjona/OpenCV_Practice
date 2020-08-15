import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

bob_image_url = "data:image/jpeg;base64,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"
urllib.request.urlretrieve(bob_image_url, "bob.png") # downloads file as "bunny.png"
img = cv2.imread('bob.png',cv2.IMREAD_GRAYSCALE)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(img, cmap='gray',interpolation='bicubic')
plt.plot([50,100],[80,100],'c',linewidth=5)
plt.show()

cv2.imwrite('SpongeBob.png',img)



