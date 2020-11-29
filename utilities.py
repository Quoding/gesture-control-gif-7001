# Imports
import cv2
from sklearn.metrics import pairwise
import numpy as np
import os

# Globals
top, left, bottom, right = 0, 0, 300, 300
KERNEL = np.ones((5, 5), np.uint8)


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

    # Blur to keep low frequencies
    roi_gray = cv2.GaussianBlur(roi_gray, (7, 7), 0)

    return frame, roi, roi_gray


def identify_hand(image, bg, threshold):
    """Locate the contours of the hand

    Args:
        image (array-like): image of interest grayscaled
        bg (array-like): Background of image
        threshold (float): threshold for countours

    Returns:
        array-like, array-like : The largest contour in the ROI, thresholded hand
    """
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(
        thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # return None, if no contours detected
    if len(cnts) == 0:
        return None, None
    else:
        # based on contour area, get the maximum contour which is the hand
        hand = max(cnts, key=cv2.contourArea)
        return hand, thresholded


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


def count_fingers(hand_contour, thresholded):
    """ Computes the numbers of finger raised in the hand contour provided
    

    Args:
        hand_contour (array-like): numpy array of points  contouring the hand
        thresholded (array-like): image thresholded of hand

    Returns:
        (int, array-like): number of fingers raised, convex hull

    """
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(hand_contour)

    # find the most extreme points in the convex hull
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the hull
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distances = pairwise.euclidean_distances(
        [(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom]
    )[0]
    max_distance = distances[distances.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * max_distance)

    # initisalize circle
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # bitwise and of thresholded hand and circle
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    (cnts, _) = cv2.findContours(
        circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    return len(cnts) - 1, chull


def apply_skin_mask(image, lower_bound=[0, 48, 80], upper_bound=[20, 255, 255]):
    """Applies a skin mask filter to the image, extracting only regions having a skin color corresponding to the skin mask

    Args:
        image (array-like): image to apply skin mask to

    Returns:
        array-like: image with skin mask applied
    """
    # Skin mask code to detect skin color and basically boost it
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_bound, dtype="uint8")
    upper = np.array(upper_bound, dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2, 2))

    # Apply treshold on skin mask to extract skin colored objects
    ret, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    # Tried to apply opening and closing but it makes the image more grainy, leaving details out and it messes up the model

    return thresholded


def get_calibration_params(pattern_size=(6, 9)):
    """Fetches calibration parameters for camera given a set of picture of a calibration target located in ./calib_pics/".

    Args:
        pattern_size (tuple, optional): Pattern size according to cv2.findChessboardCorner. Defaults to (6, 9).

    Returns:
        array-like: output of cv2.calibrateCamera(), parameters of camera calibration
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    h, w = pattern_size
    objp = np.zeros((h * w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    filenames = os.listdir("calib_pics")
    for filename in filenames:
        filename = "calib_pics/{}".format(filename)
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(gray, pattern_size)
        if retval:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    if not imgpoints:
        print(
            "Calibration could not be done. Perhaps no pictures were in the ./calib_pics/ ?"
        )
        return False, False, False, False, False
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return ret, mtx, dist, rvecs, tvecs
