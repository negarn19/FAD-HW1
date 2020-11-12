import numpy as np
import cv2 as cv

img_path1 = "C:/Negar/Computer_vision/session2/mask_flower.jpeg"
img_mask = cv.imread(img_path1)

webcam  = cv.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output_task5_video.avi',fourcc, 5.0, (640,480))

while(webcam .isOpened()):
    ret, frame = webcam.read()
    if ret==True:
        #frame = cv.flip(frame,0)

        # write the flipped frame
        out.write(frame)
        webcam_img = frame[60:419, 0:639]

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
        img_mask1[img_mask1 > 128] = 255
        img_mask1[img_mask1 <= 128] = 0

        img_mask1[img_mask1 == 255] = 1

        # multipaly images

        img_b = img_mask1 * b
        img_g = img_mask1 * g
        img_r = img_mask1 * r

        merge_final = cv.merge((img_b, img_g, img_r))
        cv.imshow('frame', merge_final)


        if cv.waitKey(1) & 0xFF == ord('q'):


            break
    else:
        break

# Release everything if job is finished
webcam.release()
out.release()
cv.destroyAllWindows()