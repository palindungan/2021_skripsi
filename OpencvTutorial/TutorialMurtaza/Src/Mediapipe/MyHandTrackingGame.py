import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)

# time to count fps
pTime = 0  # previous
cTime = 0  # current

# create an object from class
detector = htm.HandDetector()

while True:
    success, img = cap.read()  # read the image
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        # test
        print(lmList[4])

    # counting fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # drawing and show img
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv2.imshow('Image', img)

    cv2.waitKey(1)
