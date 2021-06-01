import cv2
import pickle
import numpy as np
from TutorialMurtaza.Util import BaseFunction
from tensorflow import keras

########################
width = 640
height = 480
threshold = 0.70
########################

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# pickle_in = open(BaseFunction.getBaseUrl() + '/TutorialMurtaza/Resources/model_trained.p', 'rb')
# model = pickle.load(pickle_in)

model = keras.models.load_model(BaseFunction.getBaseUrl() + '/TutorialMurtaza/Resources/model_trained.h5')


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to GRAY
    img = cv2.equalizeHist(img)  # function to balance the histogram (contras effect)
    img = img / 255  # from 0,1,2,3 ... 254,255 to 0,1
    return img


while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    # cv2.imshow('Processed Image', img)
    img = img.reshape(1, 32, 32, 1)

    # Predict
    classIndex = int(model.predict_classes(img))
    # print(classIndex)
    predictions = model.predict(img)
    # print(predictions)
    proVal = np.amax(predictions)
    print(classIndex, proVal)

    if proVal >= threshold:
        cv2.putText(imgOriginal, str(classIndex) + ', ' + str(proVal), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255), 1)

    cv2.imshow('Result', imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
