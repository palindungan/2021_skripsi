# import library
import cv2

# read image
path = "Resources/lena.jpg"
img = cv2.imread(path)
print(img.shape)  # : (512, 512, 3) -> (y,x,channel)

# resize image
width, height = 200, 400
imgResize = cv2.resize(img, (width, height))
print(imgResize.shape)  # : (400, 200, 3) -> (y,x,channel)

# crop image in matrix y,x
imgCropped = img[200:400, 100:300]
print(imgCropped.shape)  # : (200, 200) -> (y,x)
imgCropResized = cv2.resize(imgCropped, (imgResize.shape[1], imgResize.shape[0]))
print(imgCropResized.shape)  # : (400, 200, 3) -> (y,x,channel)

# show image in window
cv2.imshow("Img Lena", img)
cv2.imshow("Img Resize", imgResize)
cv2.imshow("Img Cropped", imgCropped)
cv2.imshow("Img Cropped Resized", imgCropResized)

# delay
cv2.waitKey(0)
