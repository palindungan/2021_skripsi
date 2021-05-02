import cv2
import numpy as np

# read image
path1 = "Resources/lena.jpg"
path2 = "Resources/lena.jpg"
img1 = cv2.imread(path1)
img2 = cv2.imread(path2, 0)

# resize image to x0.5
size = (0, 0)
fx = 0.5
fy = 0.5
img1 = cv2.resize(img1, size, None, fx, fy)
img2 = cv2.resize(img2, size, None, fx, fy)

# if 1 channel : convert to 3 channel / RGB
# img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

horStack = np.hstack((img1, img2))
verStack = np.vstack((img1, img2))

# show image
cv2.imshow("Hor", horStack)
cv2.imshow("Ver", verStack)

cv2.waitKey(0)
