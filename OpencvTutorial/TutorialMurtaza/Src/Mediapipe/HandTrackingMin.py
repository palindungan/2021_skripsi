import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands  # declaration before using mediapipe
hands = mpHands.Hands()  # module for hand tracking and detection
mpDraw = mp.solutions.drawing_utils  # module for drawing landmark connection

# time to count fps
pTime = 0  # previous
cTime = 0  # current

while True:
    success, img = cap.read()  # read the image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR -> RGB

    results = hands.process(imgRGB)  # preform the hand detection
    # print(results.multi_hand_landmarks)

    # detect if there is hand or not
    if results.multi_hand_landmarks:
        # detect multiple hands
        for handLms in results.multi_hand_landmarks:
            # detect index ,position (ratio) landmark  in image
            for id, lm in enumerate(handLms.landmark):
                
                print(id, lm)

            # drawing connection landmark
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # counting fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # drawing and show img
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv2.imshow('Image', img)

    cv2.waitKey(1)
