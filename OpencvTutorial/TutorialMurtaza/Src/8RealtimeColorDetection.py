# import library
import cv2
import numpy as np
from TutorialMurtaza.Util import BaseFunction

# init
frameWidth = 640
frameHeight = 480

# init video capture (webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP D SHOW = DirectShow (via videoInput)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# init window and trackbar
hsvWinName = "HSV"
hueMinTrackName = "Hue Min"
hueMaxTrackName = "Hue Max"
satMinTrackName = "Sat Min"
satMaxTrackName = "Sat Max"
valMinTrackName = "Val Min"
valMaxTrackName = "Val Max"

cv2.namedWindow(hsvWinName)
cv2.resizeWindow(hsvWinName, 640, 240)
cv2.createTrackbar(hueMinTrackName, hsvWinName, 0, 179, BaseFunction.empty)
cv2.createTrackbar(hueMaxTrackName, hsvWinName, 179, 179, BaseFunction.empty)
cv2.createTrackbar(satMinTrackName, hsvWinName, 0, 255, BaseFunction.empty)
cv2.createTrackbar(satMaxTrackName, hsvWinName, 255, 255, BaseFunction.empty)
cv2.createTrackbar(valMinTrackName, hsvWinName, 0, 255, BaseFunction.empty)
cv2.createTrackbar(valMaxTrackName, hsvWinName, 255, 255, BaseFunction.empty)

# iteration
while True:
    _, img = cap.read()  # read the image in webcam

    # resize image
    width, height = 240, 240
    img = cv2.resize(img, (width, height))

    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert color BGR -> HSV

    # get value trackbar
    hMin = cv2.getTrackbarPos(hueMinTrackName, hsvWinName)
    hMax = cv2.getTrackbarPos(hueMaxTrackName, hsvWinName)
    sMin = cv2.getTrackbarPos(satMinTrackName, hsvWinName)
    sMax = cv2.getTrackbarPos(satMaxTrackName, hsvWinName)
    vMin = cv2.getTrackbarPos(valMinTrackName, hsvWinName)
    vMax = cv2.getTrackbarPos(valMaxTrackName, hsvWinName)

    # create array to become threshold
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    mask = cv2.inRange(hsvImg, lower, upper)  # create masking
    result = cv2.bitwise_and(img, img, mask=mask)

    # stacking image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stackedImg = np.hstack([img, mask, result])

    # show image
    # cv2.imshow("Original Image", img)
    # cv2.imshow("HSV color Space", hsvImg)
    # cv2.imshow("Masking", mask)
    # cv2.imshow("Result", result)
    cv2.imshow("Stacked Image", stackedImg)

    # if press q then quit loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()  # Closes video file or capturing device.
cv2.destroyAllWindows()  # close all opened windows
