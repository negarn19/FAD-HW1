import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# load mask

img_path1 = "C:/Negar/Computer_vision/session2/mask_flower.jpeg"
img_mask = cv.imread(img_path1)


# load webcam image
webcam = cv.VideoCapture(0)
ret, frame = webcam.read()
print(ret)
webcam.release()
#cv.imshow("my image", frame)
#cv.waitKey(0)
cv.destroyAllWindows()

webcam_img = frame [60:419,0:639]






# size of image
w = img_mask.shape[0]
h = img_mask.shape[1]
dim = (h, w)

# change the size of mask
webcam_img1 = cv.resize(webcam_img, dim)

b, g, r = cv.split(webcam_img1)

# convert to gray
img_mask1 = cv.cvtColor(img_mask, cv.COLOR_BGR2GRAY)


# threshold
img_mask1 [img_mask1 > 128] = 255
img_mask1 [img_mask1 <= 128] = 0

img_mask1[img_mask1 == 255] = 1

# multipaly images

img_b = img_mask1 * b
img_g = img_mask1 * g
img_r = img_mask1 * r

merg_final = cv.merge((img_b, img_g, img_r))


# plot image mask
cv.imshow("output", merg_final)
cv.waitKey(0)


# save image
cv.imwrite ('C:/Negar/Computer_vision/session2/output_task5.jpg', merg_final)
