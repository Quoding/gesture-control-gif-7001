# Imports
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

from utilities import *

# importing vlc module 
import vlc 
# importing pafy module 
import pafy 
import youtube_dl
  
# url of the video 
url = "https://www.youtube.com/watch?v=_Eh7SaexZnI"

# Background

# Load model from https://www.kaggle.com/suhasrao/handgesturerecognition-with-99-accuracy?select=handgesturerecog_model.h5
model = load_model("model/handgesturerecog_model.h5")
class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
identified = ""

camera = cv2.VideoCapture(0)

# Load model from https://www.kaggle.com/suhasrao/handgesturerecognition-with-99-accuracy?select=handgesturerecog_model.h5
model = load_model("model/handgesturerecog_model.h5")

# Keep constant exposure
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

  
# creating pafy object of the video 
video = pafy.new(url) 
volume = 50
  
# getting best stream 
best = video.getbest() 
  
# creating vlc media player object 
media = vlc.MediaPlayer(best.url) 
  
# start playing video 
media.play() 

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
            gesture = max(set(preds), key=preds.count)
            print("Model predicted: " + str(gesture))
            identified = class_names[gesture]
            if gesture == 5: #thumb
                media.pause()
            elif gesture == 7: #ok
                media.play()
            elif gesture == 8: #palm_moved
                volume = min(volume + 10,100)
                media.audio_set_volume(volume)
            elif gesture == 3: #fist
                volume = max(0, volume-10)
                media.audio_set_volume(volume)
                    
        else:
            identified = ""
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
            if hand is not None:
                if cv2.countNonZero(hand[1]) == 0:
                    hand = None
                else:
                    hand = cv2.bitwise_and(roi_gray,roi_gray,mask = hand[1])

        if hand is not None:
            cv2.imshow("mask", hand)

            dim = (128, 128)
            hand = cv2.resize(hand, dim)
            hand = np.expand_dims(hand, axis=2)
            hand = np.expand_dims(hand, axis=0)
            preds.append(np.argmax(model.predict(hand)))

        

    # Draw rectangle on image
    cv2.rectangle(frame, (top, left), (bottom, right), (255, 0, 0), 1)
    cv2.putText(
            frame,
            identified,
            (200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            4,
            )

    # update number of frames
    num_frames += 1
    bg_frames += 1

    # Draw contours over the ROI
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyAllWindows()
camera.release()
media.stop()
