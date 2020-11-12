import cv2 as cv
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


from PIL import Image, ImageChops

img_path = "C:/Negar/Computer_vision/session1/8cef236c1c21c4.jpg"  # r" "

img = cv.imread(img_path)
# cv.imshow("my image", img)
#cv.waitKey(0)


print (len(img))
print(img.shape)
w = img.shape[0]
h = img.shape[1]

#for i in range (0,h):
   # for j in range (0,w):
     #   print (img[i,j])
# convert gray to black white


img_gray_new = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
img_gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )


#img_gray_new = img_gray/ 255 ;

img_gray [img_gray > 120] = 0
img_gray [img_gray < 120] = 1


##### image_gray_mask = ImageChops.multiply (img_gray,img_gray_new)



#### img_gray_mask = img_gray * img_gray_new

plt.imshow(img)
plt.show()

#cv.imshow("image gray", img_gray)
#cv.waitKey(0)
















