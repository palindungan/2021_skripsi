# start of import library
import cv2
import numpy as np
from TutorialMurtaza.Util import BaseFunction

# declaration
frameWidth = 640
frameHeight = 480

# capture video (in spesific folder)
# cap = cv2.VideoCapture("Resources/testVideo1.mp4")

# capture video webcam
cap = cv2.VideoCapture(0)

# setting webcam frame size
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# looping per frame video capture
while True:
    # boolean and frame video capture
    success, img = cap.read()

    # resize the frame
    img = cv2.resize(img, (frameWidth, frameHeight))

    # showing frame video
    # cv2.imshow("Video", img)

    # read image
    # img = cv2.imread("Resources/lena.jpg")

    # convert color img from BGR to GRAY
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur (src, kernel_size, sigma)
    kernelSize = (7, 7)  # odd number : rise to increase the blur
    sigmaX = 0
    imgBlur = cv2.GaussianBlur(imgGrey, kernelSize, sigmaX)

    # edge detection
    th1 = 100  # threshold dawn to increase edge
    th2 = 200  # threshold dawn to increase edge
    imgCanny = cv2.Canny(imgBlur, th1, th2)

    # Dilation
    kernelSize = (5, 5)
    kernel = np.ones(kernelSize, np.uint8)  # is matrix in odd number
    iteration = 1  # rise to increase dilation
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=iteration)

    # erosion
    iteration = 1  # rise to increase erosion
    imgEroded = cv2.erode(imgDilation, kernel, iterations=iteration)

    # show image in window
    # cv2.imshow("Lena Color", img)
    # cv2.imshow("Lena Gray", imgGrey)
    # cv2.imshow("Lena Blur", imgBlur)
    # cv2.imshow("Lena Canny", imgCanny)
    # cv2.imshow("Lena Dilation", imgDilation)
    # cv2.imshow("Lena Eroded", imgEroded)

    blankImage = np.zeros((200, 200), np.uint8)

    stackedImages = BaseFunction.stackImages(0.5, ([img, imgGrey, imgBlur], [imgCanny, imgDilation, imgEroded]))
    cv2.imshow("Stacked Image", stackedImages)

    # delay
    # cv2.waitKey(0)

    # delay 1ms and check to quit the looping when press 'q' on keyboard
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
