# import libraries
import cv2
import numpy as np
from TutorialMurtaza.Util import BaseFunction

# init
frameWidth = 640
frameHeight = 480

# init + setting video capture (webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# init window + trackbar
winName = "Parameters"
trackNameTH1 = "Threshold 1"
trackNameTH2 = "Threshold 2"

# init parameters window trackbar
frameHeightHalf = int(frameHeight / 2)
cv2.namedWindow(winName)
cv2.resizeWindow(winName, (frameWidth, frameHeightHalf))
cv2.createTrackbar(trackNameTH1, winName, 150, 255, BaseFunction.empty)
cv2.createTrackbar(trackNameTH2, winName, 255, 255, BaseFunction.empty)

while True:
    # read image
    _, img = cap.read()

    # blur image
    kernelSize = (7, 7)
    sigmaX = 1
    imgBlurred = cv2.GaussianBlur(img, kernelSize, sigmaX)

    imgGrey = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)  # convert color from BGR to Grey Scale

    # canny edge detector
    threshold1 = cv2.getTrackbarPos(trackNameTH1, winName)
    threshold2 = cv2.getTrackbarPos(trackNameTH2, winName)
    imgCanny = cv2.Canny(imgGrey, threshold1, threshold2)

    imgBlank = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    imgStacked = BaseFunction.stackImages(0.5, (
        [img, imgBlurred, imgGrey], [imgCanny, imgBlank, imgBlank]))  # stacked images

    # show image
    cv2.imshow("Image Processing", imgStacked)

    # if to quit from looping
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
