# Imports
import cv2
import logging

# Globals
background = None
num_frames = 0
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

    # Draw rectangle on frame
    frame = cv2.rectangle(frame, (top, left), (bottom, right), (255, 0, 0), 1)

    # Define ROI
    roi = clone[top:bottom, left:right]

    # Gray-scale ROI
    roi_gray = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)

    return frame, roi, roi_gray


def get_contours(edges, frame):
    """Locate contours of the edges

    Args:
        edges (array-like): Edge detection image (Canny edge detection result)

    Returns:
        array-like: The contours in the ROI
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(edges, kernel)

    cnts, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    frame_cnts = cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)

    return frame_cnts
