# ------------- NOTES -----------------
# For the next step from 12 - end
# we're going convert into class components, so we can use it for the next project where we want to use hand detector
# Convert it into class component start from step 12


# Step 1, initialization the file (import, webcam, etc)

import cv2
import mediapipe as mp
import time

# Step 12
# Create a class, so we can make it for the next project
class handDetector():
    # Step 13 -> declare the method for initialization the class

    def __init__(self, mode=False,maxHands=2, modelComplexity=1, detectionConfidence = 0.5 , trackConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence


        # Step 13, copy the step from no 2 and 7 paste it on __init__ method, but dont forget to add self
        # Step 2.
        # Define the tracker for hands that we are going to use for tracking a hand from mediapipe
        self.mpHands = mp.solutions.hands #add self first -> so it means that we're going to declare it as global property
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackConfidence) #Add the parameter
        # self.hands = self.mpHands.Hands() #Add the parameter

        # Step 7
        self.mpDraw = mp.solutions.drawing_utils

    # Step 14. After define the method init, we're going to create method findHands for detecting the hand
    def findHands(self, frame, draw = True):


        # Step 3
        # Because the frame is BGR color, we need to convert it to RGB color because hands on MP only can work for RGB color
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 4
        # After we convert it into RGB color, we need to process it using the hands variable that already created before

        # Step 15. Dont forget to add self before -> it means we're access the global property here
        self.results = self.hands.process(frameRGB)  # Dont forget to add self

        # Step 5
        # Try to print the results, to find out what is inside the results
        # The results print, it print class mediapipe solitionOutput
        # print(results)
        # To detect if there some hands on the camera using this one
        # print(results.multi_hand_landmarks)

        # Step 6
        # If there is a hand detected on the camera
        if self.results.multi_hand_landmarks:
            # We will loop it, because maybe there is much more than 1 hand on the camera
            for handLms in self.results.multi_hand_landmarks:



                # Step 6
                # After we define the mpDraw to draw the solutions, use it to draw it
                # Step 16. Also don't forget to add self first at this
                # self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)


                # Step 18. add this one if you want to draw these
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
                # Step 8. Try to run this and then see the frame of the camera if there is red node on the hand, try it for multiple hand
                # HAND_CONNECTIONS means it will connect every node so it has a connection to each other (similar with a graph)

        # Step 21
        # Dont forget to return the frame
        return frame

    # Step 21
    # Create new method called findPosition
    def findPosition(self, frame, handNo=0, draw=True):

        # Step 22
        # Create a list that will contain the landmark position
        self.lmList = []

        # Step 22
        # Add Condition

        if self.results.multi_hand_landmarks:
            # Detect the number by create new variable called myHand
            myHand = self.results.multi_hand_landmarks[handNo]


            # Step 17. Try to comment it because we dont want it to draw it
            # Step 12
            # After to find landmark on the frame, next we need to figure out the id of the landmark
            for id, lm in enumerate(myHand.landmark): #update it to myHand
                # Try to print it, this is the example of the  what is the value of id and lm
                # 20 x: 0.5787419676780701 -> 20 is id
                # y: 0.9594467282295227
                # z: -0.01320577971637249
                # print(id, lm)
                h, w, c = frame.shape #try to get height, width, channel
                cx, cy = int(lm.x*w), int(lm.y*h)
                # Try to print it
                # If we print it, it will the display the id of every landmark to every coordinates on the frame
                # print(id, cx, cy)

                # Add to the list
                self.lmList.append([id, cx, cy])

                # id == 0 is one of the node landmark on the hand
                # if id == 0:
                if draw:
                    cv2.circle(frame, (cx,cy), 7, (255,0,0), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        # Declare a list called fingers
        fingers = []
        # List tips id berisikan landmark ujung setiap jari
        tipIds = [4,8,12,16,20]

        # Conditional for thumb
        if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)


        # Condition for 4 Fingers
        for id in range(1,5):
            if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] -2] [2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers







def main():

    # Step 9
    # Declare variable to calculate the FPS
    # Ptime means previous time and cTime means current time
    pTime = 0
    cTime = 0

    # Step 19
    # For check the result, we need to declare new object that we're going to use it, and the use the class that we're already created
    detector = handDetector()

    cap = cv2.VideoCapture(0)



    while True:
        res, frame = cap.read()

        # Step 20, after we get the frame, send the frame to method findHands that we already using it
        frame = detector.findHands(frame)


        # Step 23
        # Called the method findPosition and then store it at the list
        lmList = detector.findPosition(frame, draw= False) #You also can add another parameter while called the method
        # if len(lmList) != 0:
        #     print(lmList[0]) # 0 means is the landmark position, there is 20 landmark id

        # Step 10
        # Calculate the time for display the FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # Step 11
        # After find the FPS value we need to display it on the frame
        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Frame", frame)

        key= cv2.waitKey(1)
        if key == ord('q'):
            break;

    cap.release()
    cv2.destroyAllWindows()



# It means if we run this script
if __name__ == "__main__":
    main()

