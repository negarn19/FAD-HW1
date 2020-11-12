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
# cv.imshow("my image", frame)
# cv.waitKey(0)
cv.destroyAllWindows()

# convert rgb 2 gray
webcam_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


webcame_img2 = webcam_img.copy ()
#webcam_img3 = np.empty(shape, dtype=float, order='C')

h = webcam_img.shape[0]
w = webcam_img.shape[1]
m = 0
n = 0

webcam_img2 = np.zeros([m,n],dtype=np.uint8)


for i in range(0,h):
      if webcam_img[i,:] == all([0]):
          print(webcam_img [i,:])

          m = m+1
          n = n+1




  #row  = [row[0] for row in webcam_img]
  #webcam_img = np.delete(webcam_img, [1,2,3], axis = 1)

#webcam_img1 = webcam_img.crop ((59,0,419,639))



webcam_img1 = webcam_img [60:419,0:639]
h = webcam_img1.shape[0]
w = webcam_img1.shape[1]

#webcam_img1[webcam_img1 == 0] = 1

# size of image
width = int(img_mask.shape[1])
height = int(img_mask.shape[0])
dim = (width, height)

# change the size of mask
webcam_img2 = cv.resize(webcam_img1, dim)

# convert to gray
img_mask1 = cv.cvtColor(img_mask, cv.COLOR_BGR2GRAY)


# threshold
img_mask1 [img_mask1 > 128] = 255
img_mask1 [img_mask1 <= 128] = 0

img_mask1[img_mask1 == 255] = 1



# multipaly images

img_final = webcam_img2*img_mask1


# plot image mask
cv.imshow("output",webcam_img)
cv.waitKey(0)


# save image
cv.imwrite ('C:/Negar/Computer_vision/session2/output_task4.jpg', img_final)
