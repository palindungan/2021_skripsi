# start of import library
import cv2
import numpy as np

# read image
img = cv2.imread("../Resources/lena.jpg")

# convert color img from BGR to GRAY
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur (src, kernel_size, sigma)
kernelSize = (7, 7)  # odd number : rise to increase the blur
sigmaX = 0
imgBlur = cv2.GaussianBlur(imgGrey, kernelSize, sigmaX)

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

# show image in window
cv2.imshow("Lena Color", img)
cv2.imshow("Lena Gray", imgGrey)
cv2.imshow("Lena Blur", imgBlur)
cv2.imshow("Lena Canny", imgCanny)
cv2.imshow("Lena Dilation", imgDilation)
cv2.imshow("Lena Eroded", imgEroded)

# delay
cv2.waitKey(0)
