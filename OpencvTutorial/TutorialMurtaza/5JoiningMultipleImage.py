# start of import library
import cv2
import numpy as np


def stackImages(imgArray, scale, lables=[]):
    sizeW = imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


# read image
img = cv2.imread("Resources/lena.jpg")

# convert color img from BGR to GRAY
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur (src, kernel_size, sigma)
kernelSize = (7, 7)  # odd number : rise to increase the blur
sigmaX = 0
imgBlur = cv2.GaussianBlur(imgGray, kernelSize, sigmaX)

# edge detection
th1 = 100  # threshold dawn to increase edge
th2 = 200  # threshold dawn to increase edge
imgCanny = cv2.Canny(imgBlur, th1, th2)

# Dilation
kernelSize = (5, 5)
kernel = np.ones(kernelSize, np.uint8)  # is matrix in odd number
iteration = 1  # rise to increase dilation
imgDilation = cv2.dilate(imgCanny, kernel, iterations=iteration)

# erosion
iteration = 1  # rise to increase erosion
imgEroded = cv2.erode(imgDilation, kernel, iterations=iteration)

# stacked image
scale = 0.9
images = ([img, imgGray, imgBlur], [imgCanny, imgDilation, imgEroded])
labels = ['1', '2']
# stackedImage = stackImages(images, scale, labels)

StackedImages = stackImages(images,scale)

# show image in window
# cv2.imshow("Lena Color", img)
# cv2.imshow("Lena Gray", imgGray)
# cv2.imshow("Lena Blur", imgBlur)
# cv2.imshow("Lena Canny", imgCanny)
# cv2.imshow("Lena Dilation", imgDilation)
# cv2.imshow("Lena Eroded", imgEroded)
cv2.imshow("Lena stackedImage", StackedImages)

# delay
cv2.waitKey(0)
