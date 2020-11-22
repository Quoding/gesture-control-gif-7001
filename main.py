# Imports
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

from utilities import *

# Background

camera = cv2.VideoCapture(0)

# Load model from https://www.kaggle.com/suhasrao/handgesturerecognition-with-99-accuracy?select=handgesturerecog_model.h5
model = load_model("model/handgesturerecog_model.h5")

# Keep constant exposure
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

if camera.isOpened():  # try to get the first frame
    rval, frame = camera.read()
else:
    rval = False

num_frames = 0
preds = []

# regrouper les classes 3 4 et 5

while rval:
    if num_frames == 60:
        print("Model predicted: " + str(max(set(preds), key=preds.count)))
        preds = []
        num_frames = 0

    rval, frame = camera.read()

    frame, roi, roi_gray = make_frame_roi(frame)

    skin_roi = apply_skin_mask(roi)
    cv2.imshow("skin", skin_roi)

    dim = (128, 128)
    skin_roi = cv2.resize(skin_roi, dim)
    skin_roi = np.expand_dims(skin_roi, axis=2)
    skin_roi = np.expand_dims(skin_roi, axis=0)
    preds.append(np.argmax(model.predict(skin_roi)))

    # Draw rectangle on image
    cv2.rectangle(frame, (top, left), (bottom, right), (255, 0, 0), 1)

    # update number of frames
    num_frames += 1

    # Draw contours over the ROI
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyAllWindows()
camera.release()
