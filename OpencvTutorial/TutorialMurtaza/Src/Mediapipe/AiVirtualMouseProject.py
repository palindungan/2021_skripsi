import cv2
import numpy as np
import HandTrackingModule as htm
import time
from TutorialMurtaza.Util import BaseFunction

import pyautogui
import sys

################
wCam, hCam = 640, 480
################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(maxHands=1)

pTime = 0

while True:
    # 1. Find the hand landmark
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index, and middle finger

    # 3. check which finger are up

    # 4. Only index finger : moving mode

    # 5. Convert the coordinate

    # 6. Smoothen values

    # 7. Move mouse

    # 8. Both Index and middle finger are up : Clicking Mode

    # 9. Find distance between fingers

    # 10. click mouse if distance short

    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow('Image', img)
    cv2.waitKey(1)
