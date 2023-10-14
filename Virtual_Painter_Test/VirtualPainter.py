import cv2
import numpy as np
import time
import os

import HandTrackingModule as htm
detector = htm.handDetector(detectionConfidence = 0.85)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

myListDirectory = os.listdir("header")
print(myListDirectory)
overlayList = []
for imPath in myListDirectory:
    image = cv2.imread(f'header/{imPath}')
    overlayList.append(image)
header = overlayList[0]
drawColor = (0,0,255)
brushThickness = 7
eraserThickness = 40
xp, yp = 0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8) 



while True:
    res, frame = cap.read()
    frame = cv2.flip(frame, 1) 
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=True)
    # print(lmList)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)
        fingers = detector.fingersUp()
        # print(fingers)
        if fingers[1] and fingers[2]:
            xp, yp = 0,0
            print("Selection mode")
            cv2.rectangle(frame, (x1,y1-25), (x2, y2+25), 
            drawColor, cv2.FILLED)
            if y1<125:
                if 320 < x1 < 480:
                    header = overlayList[0]
                    drawColor = (0,0,255)
                elif 480 < x1 < 630:
                    header = overlayList[1]
                    drawColor = (0,255,0)
                elif 630 < x1 < 840:
                    header = overlayList[2]
                    drawColor = (255,0,0)
                elif x1> 1000:
                    header = overlayList[3]
                    drawColor = (0,0,0)
        
        if fingers[1] and fingers[2] == False:
            print("Drawing Mode")
            cv2.circle(frame, (x1, y1), 15, drawColor, cv2. FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0,0,0):
                cv2.line(frame, (xp, yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), drawColor, eraserThickness)
            else:
                cv2.line(frame, (xp, yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), drawColor, brushThickness)
            xp, yp = x1, y1


    frameGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, frameInvers = cv2.threshold(frameGray, 50, 255, cv2. 
    THRESH_BINARY_INV)
    frameInvers = cv2.cvtColor(frameInvers, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, frameInvers)
    frame = cv2.bitwise_or(frame, imgCanvas)
    frame[0:125, 0:1280] = header
    cv2.imshow("Frame", frame) 
    cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break   
cap.release()
cv2.destroyAllWindows()