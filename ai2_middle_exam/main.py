import cv2
import mediapipe as mp
from math import hypot

cap = cv2.VideoCapture(0)
mouth_img = cv2.imread('mouth.png')
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=4)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            leftx = 0
            lefty = 0
            rightx = 0
            righty = 0
            centerx = 0
            centery = 0
            for lm_id, lm in enumerate(face_landmarks.landmark):
                h, w, c = rgb.shape
                x, y = int(lm.x * w), int(lm.y * h)
                if lm_id == 91:
                    leftx, lefty = x, y
                if lm_id == 128:
                    rightx, righty = x, y
                if lm_id == 14:
                    centerx, centery = x, y
            mouth_width = int(hypot(rightx - leftx, righty - lefty * 1.2))
            mouth_height = int(mouth_width * 0.8)
            if (mouth_width and mouth_height) != 0:
                dog_mouth = cv2.resize(mouth_img, (mouth_width, mouth_height))
            top_left = (int(centerx - mouth_width/2),
                        int(centery - mouth_height/2))
            bottom_right = (int(centerx+mouth_width/2),
                            int(centery+mouth_height/2))
            mouth_area = frame[top_left[1]: top_left[1] +
                               mouth_height, top_left[0]: top_left[0]+mouth_width]
            dog_mouth_gray = cv2.cvtColor(dog_mouth, cv2.COLOR_BGR2GRAY)
            _, dog_mask = cv2.threshold(
                dog_mouth_gray, 25, 255, cv2.THRESH_BINARY_INV)
            try:
                no_mouth = cv2.bitwise_and(mouth_area, mouth_area, mask=dog_mask)
                final_mouth = cv2.add(no_mouth, dog_mouth)
                frame[top_left[1]: top_left[1] + mouth_height,
                    top_left[0]: top_left[0] + mouth_width] = final_mouth
            except:
                pass

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow()
