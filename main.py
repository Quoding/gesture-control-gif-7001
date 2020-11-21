# Imports
import cv2
import imutils
import numpy as np

from utilities import *

#Background
bg = None 

cv2.namedWindow("Camera Feed")

camera = cv2.VideoCapture(0)

#Keep constant exposure
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

if camera.isOpened():  # try to get the first frame
    rval, frame = camera.read()
else:
    rval = False

num_frames = 0

while rval:
    rval, frame = camera.read()

    frame, roi, roi_gray = make_frame_roi(frame)
    
    if num_frames < 30:
            bg = average(roi_gray, bg, 0.5)
    else:
    
        hand = identifyhand(roi_gray, bg, 25)

        if hand is not None:
            
            # find the convex hull of the segmented hand region
            chull = cv2.convexHull(hand)
            
            # draw the segmented region and display the frame
            cv2.drawContours(frame, [chull], -1, (0, 0, 255))

    # Draw rectangle on image
    cv2.rectangle(frame, (top, left), (bottom, right), (255, 0, 0), 1)
    
    #update number of frames
    num_frames += 1
    
    # Draw contours over the ROI
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    if key == 114: #Reload background on r
        bg = None
        num_frames = 0

cv2.destroyAllWindows()
camera.release()
