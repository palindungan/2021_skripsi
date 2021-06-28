import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Setting
##########################
wCam, hCam = 1080, 720  # width and height
##########################

# Cam Capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0  # previous time -> for fps

detector = htm.HandDetector(detectionCon=0.7, maxHands=1)  # import hand tracking module

# Start of sound speaker API
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volumeBar = 400
volPer = 0
# print(volRange)
# End of sound speaker API

area = 0
colorVol = (255, 255, 0)

while True:
    success, img = cap.read()

    # Find hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)

    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        # Filter Based on size / Normalization (500px)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        # print(area)
        if 250 < area < 1000:

            # Find Distance between index and Thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)

            # Convert Volume
            # pixel range = 50 - 300
            # volume range = -65 - 0
            volumeBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])

            # Reduce Resolution to make it smoother / not change when detect very small distance
            smoothness = 10
            volPer = smoothness * round(volPer / smoothness)

            # Check Finger Up
            fingers = detector.fingersUp()
            print(fingers)

            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 255, 0)

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volumeBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set : {int(cVol)}', (400, 70), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)

    cv2.imshow('Image', img)

    # delay 1ms and check to quit the looping when press 'q' on keyboard
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
