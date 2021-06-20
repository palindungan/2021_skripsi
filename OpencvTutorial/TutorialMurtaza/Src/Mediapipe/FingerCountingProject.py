import cv2
import os
import time
from TutorialMurtaza.Util import BaseFunction

#############
wCam, hCam = 640, 480
#############

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = BaseFunction.getBaseUrl() + '/TutorialMurtaza/Resources/hand_counting'
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

while True:
    success, img = cap.read()

    # img[0:y, 0:x] img[height, width] image size is 200
    img[0:200, 0:200] = overlayList[0]

    cv2.imshow('Image', img)

    cv2.waitKey(1)
