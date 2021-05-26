import cv2
from TutorialMurtaza.Util import BaseFunction

###############################
path = BaseFunction.getBaseUrl() + '/TutorialMurtaza/Resources/haar/classifier/cascade.xml'
cameraNo = 1
objectName = 'Spoon'
frameWidth = 640
frameHeight = 480
color = (255, 0, 255)
#################################

# Camera Parameter
cap = cv2.VideoCapture(cameraNo)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# Create Trackbar
winNameResult = 'result'
trackbarNameScale = 'Scale'
trackbarNameNeighbor = 'Neighbor'
trackbarNameMinArea = 'Min Area'
trackbarNameBrightness = 'Brightness'

cv2.namedWindow(winNameResult)
cv2.resizeWindow(winNameResult, frameWidth, frameHeight + 100)
cv2.createTrackbar(trackbarNameScale, winNameResult, 400, 1000, BaseFunction.empty)
cv2.createTrackbar(trackbarNameNeighbor, winNameResult, 8, 20, BaseFunction.empty)
cv2.createTrackbar(trackbarNameMinArea, winNameResult, 0, 100000, BaseFunction.empty)
cv2.createTrackbar(trackbarNameBrightness, winNameResult, 180, 255, BaseFunction.empty)

# LOAD THE CLASSIFIER DOWNLOADED
cascade = cv2.CascadeClassifier(path)

print(cascade)
print(path)

while True:
    # set camera brightness from trackbar value
    cameraBrightness = cv2.getTrackbarPos(trackbarNameBrightness, winNameResult)
    cap.set(10, cameraBrightness)

    # get camera image and convert to greyscale
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # DETECT THE OBJECT USING THE CASCADE
    scaleVal = 1 + (cv2.getTrackbarPos(trackbarNameScale, winNameResult) / 1000)
    neighborVal = cv2.getTrackbarPos(trackbarNameNeighbor, winNameResult)
    objects = cascade.detectMultiScale(imgGray, scaleVal, neighborVal)

    # Display Detected Objects
    for (x, y, w, h) in objects:
        area = w * h
        minAreaVal = cv2.getTrackbarPos(trackbarNameMinArea, winNameResult)
        if area > minAreaVal:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, objectName, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            roiColor = img[y:y + h, x:x + w]

    cv2.imshow(winNameResult, img)

    # if to quit from looping
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
