import cv2
import os
import time
from TutorialMurtaza.Util import BaseFunction

#############################
myPath = BaseFunction.getBaseUrl() + '/TutorialMurtaza/Resources/dataset/images'  # PATH TO SAVE IMAGE
cameraNo = 1
cameraBrightness = 190
moduleVal = 10  # SAVE EVERY 1 FRAME TO AVOID REPETITION
minBlur = 500  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
grayImage = False  # IMAGE SAVED COLORED OR GRAY
saveData = True  # SAVE DATA FLAG
showImage = True  # IMAGE DISPLAY FLAG
imgWidth = 180
imgHeight = 120

#############################

global countFolder
count = 0
countSave = 0

cap = cv2.VideoCapture(cameraNo)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, cameraBrightness)


# make folder
def saveDataFunc():
    global countFolder
    countFolder = 0

    while os.path.exists(myPath + str(countFolder)):
        countFolder = countFolder + 1

    os.makedirs(myPath + str(countFolder))


if saveData:
    saveDataFunc()

while True:
    success, img = cap.read()
    imgCopy = img.copy()
    img = cv2.resize(img, (imgWidth, imgHeight))

    if grayImage:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if saveData:
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        if count % moduleVal == 0 and blur > minBlur:
            nowTime = time.time()
            cv2.imwrite(
                myPath +
                str(countFolder) +
                '/' +
                str(countSave) +
                '_' +
                str(int(blur)) +
                '_' +
                str(nowTime) + '.png',
                img)
            countSave = countSave + 1
        count = count + 1

    if showImage:
        cv2.imshow('Image', imgCopy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
