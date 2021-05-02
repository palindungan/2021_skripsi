# import library
import cv2
import numpy as np

# create matrix image
height, width = 200, 400
img = np.zeros((height, width, 3), np.uint8)  # black image y,x : 0-255 int (np.uint8)
print(img)

img[:] = 0, 255, 0  # select (:) or all and convert BGR value to specific value (blue image)
img[0:50, 0:400] = 255, 0, 0  # select specific pixel and change value matrix

# create line
point1 = (0, 0)  # x,y
point2 = (100, 200)  # x,y
color = (0, 0, 255)
cv2.line(img, point1, point2, color)  # create line

# draw the rectangle
point1 = (0, 0)  # x,y top left
point2 = (150, 100)  # x,y bottom right
# thickness = 2
thickness = cv2.FILLED  # fill the area
cv2.rectangle(img, point1, point2, color, thickness)  # draw the rectangle

# draw circle
centerPoint = (100, 50)  # x,y
radius = 50
color = (255, 255, 0)
thickness = cv2.FILLED  # fill the area
cv2.circle(img, centerPoint, radius, color, thickness)  # draw circle

# draw text
text = "Draw Text"
position = (100, 50)  # x,y
font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 1
color = (0, 150, 0)
thickness = 2
cv2.putText(img, text, position, font, fontScale, color, thickness)  # draw text

cv2.imshow("zero", img)

cv2.waitKey(0)
