# Imports
import argparse
from utilities import *

import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

media = setup_video()
identified = ""
bg_frames = 0
wait_frames = 0
preds = []
bg = None
class_names = [
    "down",
    "palm",
    "l",
    "fist",
    "fist_moved",
    "thumb",
    "index",
    "ok",
    "palm_moved",
    "c",
]
volume = 50
media.audio_set_volume(volume)
USE_SKIN = get_args()
# Load model from https://www.kaggle.com/suhasrao/handgesturerecognition-with-99-accuracy?select=handgesturerecog_model.h5
model = load_model("model/handgesturerecog_model.h5")

camera = cv2.VideoCapture(0)
# Keep constant exposure
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

if camera.isOpened():  # try to get the first frame
    rval, frame = camera.read()
else:
    rval = False

# Compute background for first 30 frames
while bg_frames < 30 and not USE_SKIN:
    rval, frame = camera.read()

    frame, roi, roi_gray = make_frame_roi(frame)

    bg = average(roi_gray, bg, 0.5)

    bg_frames += 1


while rval:
    if len(preds) >= 60 and wait_frames <= 0:
        gesture = max(set(preds), key=preds.count)
        identified = class_names[gesture]
        print(identified)
        volume = do_action(gesture, media, volume)
        wait_frames = 30
        preds = []

    elif wait_frames <= 0:
        identified = ""

    wait_frames -= 1

    rval, frame = camera.read()

    frame, roi, roi_gray = make_frame_roi(frame)

    if USE_SKIN:
        skinmask = apply_skin_mask(roi)
        hand = cv2.bitwise_and(roi_gray, roi_gray, mask=skinmask)

    else:
        hand, thresholded = identify_hand(roi_gray, bg, 25)
        if hand is not None:
            if cv2.countNonZero(thresholded) == 0:
                hand = None
            else:
                hand = cv2.bitwise_and(roi_gray, roi_gray, mask=thresholded)

    if hand is not None:
        cv2.imshow("mask", hand)
        if wait_frames <= 0:
            dim = (128, 128)
            hand = cv2.resize(hand, dim)
            hand = np.expand_dims(hand, axis=2)
            hand = np.expand_dims(hand, axis=0)
            pred = np.argmax(model.predict(hand))
            # Do not append classes which are problematic
            if pred not in [0, 7]:
                preds.append(pred)

    # Draw rectangle on image
    cv2.rectangle(frame, (top, left), (bottom, right), (255, 0, 0), 1)
    cv2.putText(
        frame, identified, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4,
    )

    # Draw contours over the ROI
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1)
    if key == 27:  # exit on ESC
        break

cv2.destroyAllWindows()
camera.release()
media.stop()
