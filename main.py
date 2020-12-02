# Imports
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

from utilities import *

# Background
bg = None

cv2.namedWindow("Camera Feed")

camera = cv2.VideoCapture(0)

# Keep constant exposure
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

if camera.isOpened():  # try to get the first frame
    rval, frame = camera.read()
else:
    rval = False

num_frames = 0
bg_frames = 0
preds = []
bg = None

use_skin = False

# regrouper les classes 3 4 et 5

while rval:
    if num_frames == 30:
        if len(preds) > 0:
            print("Model predicted: " + str(max(set(preds), key=preds.count)))
            cv2.putText(
            frame,
            str(num),
            (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            4,
            )
        preds = []
        num_frames = 0
            

    rval, frame = camera.read()

    frame, roi, roi_gray = make_frame_roi(frame)
    
    if bg_frames < 30:
        bg = average(roi_gray, bg, 0.5)
    else:
        if use_skin:
            hand = apply_skin_mask(roi)
        else:
            hand = identify_hand(roi_gray, bg, 25)
            #if cv2.countNonZero(hand[1]) == 0:
              #  hand = None
           # else:
              #  hand = cv2.bitwise_and(roi_gray,roi_gray,mask = hand[1])

        if hand is not None:
            cv2.imshow("mask", hand[1])

            dim = (128, 128)
            hand = cv2.resize(hand[1], dim)
            hand = np.expand_dims(hand, axis=2)
            hand = np.expand_dims(hand, axis=0)
            preds.append(np.argmax(model.predict(hand)))

        


    # Draw rectangle on image
    cv2.rectangle(frame, (top, left), (bottom, right), (255, 0, 0), 1)

    # update number of frames
    num_frames += 1
    bg_frames += 1

    # Draw contours over the ROI
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    if key == 114:  # Reload background on r
        bg = None
        num_frames = 0

cv2.destroyAllWindows()
camera.release()
