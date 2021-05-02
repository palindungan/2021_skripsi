import cv2
import numpy as np
from TutorialMurtaza.Util import BaseFunction

path = BaseFunction.getBaseUrl() + "/TutorialMurtaza/Resources/data/cards2.png"
img = cv2.imread(path)

# point (x,y) = 235,441 ... 680,215 ... 444,680 ... 888,395
# create matrix
width, height = 500, 250
pts1 = np.float32([[235, 441], [680, 215], [444, 680], [888, 395]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))

# draw the circle on each node
for x in range(0, len(pts1)):
    cv2.circle(img, (pts1[x, 0], pts1[x, 1]), 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (pts2[x, 0], pts2[x, 1]), 5, (0, 0, 0), cv2.FILLED)

cv2.imshow("img", img)
cv2.imshow("imgOutput", imgOutput)

cv2.waitKey(0)
