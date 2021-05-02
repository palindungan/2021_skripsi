# start of import library
import cv2

# declaration
frameWidth = 640
frameHeight = 480

# capture video (in spesific folder)
cap = cv2.VideoCapture("Resources/testVideo1.mp4")

# capture video webcam
# cap = cv2.VideoCapture(0)

# setting webcam frame size
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

# looping per frame video capture
while True:
    # boolean and frame video capture
    success, img = cap.read()

    # resize the frame
    img = cv2.resize(img, (frameWidth, frameHeight))

    # showing frame video
    cv2.imshow("Video", img)

    # delay 1ms and check to quit the looping when press 'q' on keyboard
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
