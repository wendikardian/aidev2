import cv2
import numpy as np

img = cv2.imread('images/color_detection.png')
red = ([0, 0, 30], [50, 56, 255])
blue = ([30,0, 0], [255, 150, 50])
green = ([0, 30, 0], [100, 255, 100])
white = ([255, 255, 255], [255, 255, 255])
boundaries = [red,blue,green,white]
for (lower, upper) in boundaries:
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow("Color Detection", output)
    cv2.waitKey(0)
