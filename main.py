# Imports
import cv2
import imutils
import numpy as np

from utilities import *

cv2.namedWindow("Camera Feed")

camera = cv2.VideoCapture(0)

if camera.isOpened():  # try to get the first frame
    rval, frame = camera.read()
else:
    rval = False


while rval:
    rval, frame = camera.read()

    frame, roi, roi_gray = make_frame_roi(frame)

    edges = cv2.Canny(roi_gray, 100, 200)

    frame_cnts = get_contours(edges, frame)

    # Draw contours over the ROI
    cv2.imshow("Camera Feed", frame_cnts)

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("Camera Feed")
vc.release()
