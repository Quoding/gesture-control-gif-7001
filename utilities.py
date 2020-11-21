# Imports
import cv2
import logging

# Globals
background = None
top, left, bottom, right = 0, 0, 300, 300


def make_frame_roi(frame):
    """Adds rectangle to frame and creates ROIs (regular and gray-scale)

    Args:
        frame (array-like): Image / frame from camera feed

    Returns:
        (array-like, array-like, array-like) : (frame wit rectangle, roi, gray-scale roi)
    """

    # flip image so brain doesn't fart
    frame = cv2.flip(frame, 1)

    # Copy the frame so we can alter it as we want without having to deal with the rectangle
    clone = frame.copy()

    # Define ROI
    roi = clone[top:bottom, left:right]

    # Gray-scale ROI
    roi_gray = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)
    
    #Blur to keep low frequencies
    roi_gray = cv2.GaussianBlur(roi_gray, (7, 7), 0)

    return frame, roi, roi_gray


def identifyhand(image, bg, threshold):
    """Locate the contours of the hand

    Args:
        image (array-like): image of interest grayscaled
        bg (array-like): Background of image
        threshold (float): threshold for countours

    Returns:
        array-like : The largest contour in the ROI
    """
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return None
    else:
        # based on contour area, get the maximum contour which is the hand
        hand = max(cnts, key=cv2.contourArea)
        return hand

def average(image, bg, weight):
    """Adds image to the average of the background with the weight specified

    Args:
        image (array-like): Image to add to average
        bg (array-like): current background
        weight: weight of new image in background
        
    Returns:
        (array-like): the resulting bg

    """
    
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return bg

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, weight)
    
    return bg