# Start of import library
import cv2

# read image
img = cv2.imread("../Resources/lena.jpg")

# show image
cv2.imshow("Lena", img)

# delay / waiting function
cv2.waitKey(0)
