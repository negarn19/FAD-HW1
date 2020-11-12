import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# load image
img_path = "C:/Negar/Computer_vision/session1/8cef236c1c21c4.jpg"
img_main = cv.imread(img_path)

img_path3 = "C:/Negar/Computer_vision/session1/8cef236c1c21c4.jpg"
img1 = cv.imread(img_path3)


img_path2 = "C:/Negar/Computer_vision/session2/kmeans_k=3.jpg"
img = cv.imread(img_path2)

# Convert RGB to Gray

img_mask = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)




# threshold
img_mask [img_mask < 120] = 0
img_mask [img_mask >= 120] = 255

mask = cv. imread('img_mask.jpg')

w = img_mask.shape[0]
h = img_mask.shape[1]




#for i in range(0,w):
   # for j in range(0,h):
    #  if img_mask [i,j] == 255:
      #    img_gray [i,j] == 0



for i in range(w):
  for j in range(h):
      if img_mask[i,j] == 255:
        img_main[i, j] = (255, 255, 255)










#fg = cv.bitwise_or(img_mask, img_mask, mask=mask)

# get second masked value (background) mask must be inverted
#mask = cv.bitwise_not(mask)

#bk = cv.bitwise_or(img, img, mask=mask)

# combine foreground+background
#final = cv.bitwise_and(img, img_mask)


# show image
#cv.imshow("image mask", img_mask)
#cv.waitKey(0)


#subplot(r,c) provide the no. of rows and columns

plt.figure(1)
plt.subplot(131)
plt.title('input')
plt.imshow(img1)
plt.subplot(132)
plt.title('mask')
plt.imshow(img_mask, cmap=plt.cm.gray)
plt.subplot(133)
plt.title('input with mask')
plt.imshow(img_main)
plt.show()







