from PIL import Image, ImageChops
import cv2 as cv
import tensorflow as tf
import keras

import colorsys

img_path = "C:/Negar/Computer_vision/session1/8cef236c1c21c4.jpg"  # r" "

img = cv.imread(img_path)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray_new = img_gray / 255

# image_gray_mask = ImageChops.multiply (img_gray, img_gray_new)
image_gray_mask = img_gray * img_gray_new
#cv.imshow("image gray", image_gray_mask)
cv.waitKey(0)

hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

cv.imshow("image gray", img_gray_new)
cv.waitKey(0)

h,s,v = cv.split(hsv_img)
