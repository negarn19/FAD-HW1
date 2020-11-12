import numpy as np
import cv2 as cv
img = np.array ([1,2])

for i in range(1):
    for j in range(1):
        np.append (img[i,j],2)

cv.imshow(img)



