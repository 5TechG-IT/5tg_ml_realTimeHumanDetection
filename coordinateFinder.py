import cv2
import numpy as np
import imutils

# color to detect in RGB
blue = 36 
green = 28 
red = 237
 
color = np.uint8([[[blue, green, red]]])
hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
 
hue = hsv_color[0][0][0]
 
print("Lower bound is :"),
print("[" + str(hue-10) + ", 100, 100]\n")
 
print("Upper bound is :"),
print("[" + str(hue + 10) + ", 255, 255]")

img = cv2.imread('area.png', 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_range = np.array([169, 100, 100], dtype=np.uint8)
upper_range = np.array([189, 255, 255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_range, upper_range)
 
# find contours
ret, thresh = cv2.threshold(mask, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,0,0), 2)

c = max(contours, key=cv2.contourArea)

extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

cv2.circle(img, extLeft, 5, (0, 0, 255), -1)
cv2.circle(img, extRight, 5, (0, 255, 0), -1)
cv2.circle(img, extTop, 5, (255, 0, 0), -1)
cv2.circle(img, extBot, 5, (255, 255, 0), -1)

print("ext Left: ", extLeft)
print("ext right: ", extRight)
print("ext top: ", extTop)
print("ext bottom: ", extBot)

# write to images
cv2.imwrite('mask.png',mask)
cv2.imwrite('img.png',img)
