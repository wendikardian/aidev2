import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False,maxHands=2,modelComplexity=1, detectionConfidence = 0.5 , trackConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self. 
        maxHands, self.modelComplexity,self.detectionConfidence, 
        self.trackConfidence) 
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, frame, draw = True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo = 0, draw = True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark): 
                h, w, c = frame.shape 
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx,cy), 7, (255,0,0), cv2.FILLED)
        return self.lmList
    
    def fingersUp(self):
        fingers = []
        tipIds = [4,8,12,16,20]
        if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1,5):
            if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] -2] [2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

        
