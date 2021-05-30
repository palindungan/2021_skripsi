import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from TutorialMurtaza.Util import BaseFunction
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D

##########################
# SETTING
path = BaseFunction.getBaseUrl() + '/TutorialMurtaza/Resources/MyData/'
pathLabels = ''
testRatio = 0.2
valRation = 0.2
imageDimensions = (32, 32, 3)

##########################
# import dataset 0-9, create one row images array and labels array
# start
##########################


images = []
classNo = []
myList = os.listdir(path)  # get list content of folder => 0,1,2,3,4,5,6,7,8,9
noOfClasses = len(myList)  # get number of folder => 10
print('Total No of Classes Detected = ', noOfClasses)
print('Importing Classes ........')

# for in each folder 0,1,2,3,4,5,6,7,8,9
for x in range(0, noOfClasses):
    myPicList = os.listdir(path + '/' + str(x))  # get all image in class folder
    # print(myPicList)
    print(x, end=' ')

    # for in each file on folder img001-00001 ... img001-01016 --> img010-01016
    for y in myPicList:
        curImg = cv2.imread(path + '/' + str(x) + '/' + y)  # read each file image
        curImg = cv2.resize(curImg,
                            (imageDimensions[0], imageDimensions[1]))  # resize image to decrease computation cost
        images.append(curImg)  # add image matrix in list
        classNo.append(x)  # add class in List
        # print(images)

print(' ')
print('Number of Images = ' + str(len(images)))
print('Number of Classes = ' + str(len(classNo)))

# convert list to array
images = np.array(images)
classNo = np.array(classNo)

print('shape = ' + str(images.shape))  # (10160, 32, 32, 3)
print('shape = ' + str(classNo.shape))  # (10160,)

##########################
# splitting and shuffle the data
# start
##########################

# splitting data training and testing
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)

print('Split Ratio = ' + str(testRatio))  # ration

# print('X Train = ' + str(X_train.shape))  # images
# print('X Test = ' + str(X_test.shape))  # images
# print('Y Train = ' + str(y_train.shape))  # classes
# print('Y Test = ' + str(y_test.shape))  # classes

# splitting data training and validation
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRation)

print('X Training = ' + str(X_train.shape))  # images
print('Y Training = ' + str(y_train.shape))  # classes

print('X Testing = ' + str(X_test.shape))  # images
print('Y Testing = ' + str(y_test.shape))  # classes

print('X Validation = ' + str(X_validation.shape))  # images
print('Y Validation = ' + str(y_validation.shape))  # classes

##########################
# preprocessing and reshaping the data
# start
##########################

numOfSamples = []

# return count(index array where class is present)
for x in range(0, noOfClasses):
    number = len(np.where(y_train == x)[0])
    print('total class of ' + str(x) + ' is ' + str(number))
    numOfSamples.append(number)

print(numOfSamples)

plt.figure(figsize=(10, 5))  # create a figure 10*5 inch
# x,y -> x = 0,1,2,3,4,5,6,7,8,9 | y = samples [661, 653, 627, 638, 653, 666, 653, 658, 646, 647]
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title('No of Images for each Class')
plt.xlabel('Class ID')
plt.ylabel('Number of Images')
plt.show()

print('shape before = ' + str(X_train[30].shape))  # check before preProcessing


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to GRAY
    img = cv2.equalizeHist(img)  # function to balance the histogram (contras effect)
    img = img / 255  # from 0,1,2,3 ... 254,255 to 0,1
    return img


# map = processing each matrix image in a function -> add in list -> convert list to array
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# img = X_train[30]
# img = cv2.resize(img, (300, 300))
# cv2.imshow('Processed Image', img)
# cv2.waitKey(0)

print('shape after = ' + str(X_train[30].shape))

print('before reshape = ' + str(X_train.shape))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

print('after reshape = ' + str(X_train.shape))

##########################
# image augmentation
# start
##########################

# preparation for generate data
# defines the configuration for image data preparation and augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

# This will calculate any statistics required to actually perform the transforms to your image data.
# help generator to calculate some statistic before perform transformation
dataGen.fit(X_train)

##########################
# One Hot Encode (one_hot_encode)
# start
##########################

# before hot encode
print(y_train[0])

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

# after hot encode
print(y_train[0])


##########################
# Create the Model and Training
# start
##########################

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1),
                      activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = myModel()
print(model.summary())

batchSizedVal = 50
epochsVal = 10
stepsPerEpochVal = 2000

history = model.fit_generator(
    dataGen.flow(X_train, y_train, batch_size=batchSizedVal),
    steps_per_epoch=stepsPerEpochVal,
    epochs=epochsVal,
    validation_data=(X_validation, y_validation),
    shuffle=1
)

