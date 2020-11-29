import cv2

camera = cv2.VideoCapture(0)
if camera.isOpened():  # try to get the first frame
    rval, frame = camera.read()
else:
    rval = False

cnt = 0
while rval:
    rval, frame = camera.read()

    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(20)

    if key == 27:  # exit on ESC
        break
    elif key == 114:  # Reload background on r
        cv2.imwrite("calib_pics/pic_{}.png".format(cnt), frame)
        cnt += 1

