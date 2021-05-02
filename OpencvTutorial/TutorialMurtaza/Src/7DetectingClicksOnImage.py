import cv2
import numpy as np
from TutorialMurtaza.Util import BaseFunction

circles = np.zeros((4, 2), np.int)
counter = 0


def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x, y)
        circles[counter] = x, y
        counter = counter + 1
        print(circles)


path = BaseFunction.getBaseUrl() + "/TutorialMurtaza/Resources/data/cards2.png"
img = cv2.imread(path)
winNameImg = "Img"

while True:

    if counter == 4:
        # point (x,y) = 235,441 ... 680,215 ... 444,680 ... 888,395
        # create matrix
        width, height = 500, 250
        pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (width, height))
        winNameImgOutput = "ImgOutput"
        cv2.imshow(winNameImgOutput, imgOutput)

    # draw the circle on each node
    for i in range(0, len(circles)):
        cv2.circle(img, (circles[i, 0], circles[i, 1]), 5, (0, 0, 255), cv2.FILLED)

    cv2.imshow(winNameImg, img)
    cv2.setMouseCallback(winNameImg, mousePoints)
    cv2.waitKey(1)
