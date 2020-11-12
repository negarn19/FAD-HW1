import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read the image
img_path = "C:/Negar/Computer_vision/session1/8cef236c1c21c4.jpg"  # r" "

img = cv.imread(img_path)


# convert to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)





pixel_value = img.reshape((-1, 1))

pixel_value = np.float32 (pixel_value)

print(pixel_value.shape)

criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 10, 0.2)

k = 5




_, labels, (centers) = cv.kmeans(pixel_value, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()


# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]



# reshape back to the original image dimension
segmented_image = segmented_image.reshape(img.shape)



# show the image
plt.imshow(segmented_image)
plt.show()
cv.imwrite ('C:/Negar/Computer_vision/session2/kmeans_k=5.jpg', segmented_image)

#segmented_image_mask  = cv.cvtColor(segmented_image, cv.COLOR_RGB2GRAY)

#segmented_image_mask [segmented_image_mask > 128] = 0

#segmented_image_mask [segmented_image_mask <= 128] = 0

#cv.imshow("image mask" , segmented_image_mask)
# cv.waitKey(0)