import cv2
import mediapipe as mp
from math import hypot

# cap = cv2.VideoCapture('data/2.mp4')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

nose_img = cv2.imread('data/pig_nose.png')  # (860, 563) w,h ratio=563/860=0.65

# prevTime = 0

nose_landmarks = [49, 279, 5]  # 5 = center nose point

# total 468 landmarks
mpDraw = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=4)  # max_num_faces=1
# drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:

    ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 480)) #(1200, 650)
    # mediapipe needs RGB image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # draw the landmakrs
            # mpDraw.draw_landmarks(
            #     frame,
            #     face_landmarks,
            #     mpFaceMesh.FACE_CONNECTIONS,
            #     drawSpec,
            #     drawSpec
            # )
            mpDraw.draw_landmarks(image=frame,
                                  landmark_list=face_landmarks, connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_tesselation_style()
                                  )
            #Code draw_landmarks sebelumnya
            mpDraw.draw_landmarks(image=frame,
                                  landmark_list=face_landmarks,
                                  connections=mpFaceMesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_contours_style())
            # mpDraw.draw_landmarks(argument) berikan komentar
            leftnosex = 0
            leftnosey = 0
            rightnosex = 0
            rightnosey = 0
            centernosex = 0
            centernosey = 0

            # get each landmark info
            for lm_id, lm in enumerate(face_landmarks.landmark):

                # getting original value
                h, w, c = rgb.shape
                x, y = int(lm.x * w), int(lm.y * h)

                # calculating nose wid th
                if lm_id == 49:
                    leftnosex, leftnosey = x, y
                if lm_id == 279:
                    rightnosex, rightnosey = x, y
                if lm_id == 5:
                    centernosex, centernosey = x, y

                # display only nose landmarks
                # if lm_id in nose_landmarks:
                #     cv2.putText(
                #         frame,q
                #         str(lm_id),
                #         (x, y),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.3,
                #         (0,0,255),
                #         1
                #     )
            nose_width = int(hypot(rightnosex-leftnosex,
                             rightnosey-leftnosey*1.2))
            # nose_width = int(rightnosex-leftnosex*1.2);
            nose_height = int(nose_width*0.8)

            if (nose_width and nose_height) != 0:
                pig_nose = cv2.resize(nose_img, (nose_width, nose_height))

            top_left = (int(centernosex-nose_width/2),
                        int(centernosey-nose_height/2))
            bottom_right = (int(centernosex+nose_width/2),
                            int(centernosey+nose_height/2))

            nose_area = frame[
                top_left[1]: top_left[1]+nose_height,
                top_left[0]: top_left[0]+nose_width
            ]

            pig_nose_gray = cv2.cvtColor(pig_nose, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(
                pig_nose_gray, 25, 255, cv2.THRESH_BINARY_INV)
            no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(no_nose, pig_nose)
            frame[
                top_left[1]: top_left[1]+nose_height,
                top_left[0]: top_left[0]+nose_width
            ] = final_nose

    # currTime = time.time()
    # fps = 1/(currTime-prevTime)
    # prevTime = currTime

    # cv2.putText(
    #     frame,
    #     f"FPS: {str(round(fps))}",
    #     (20, 70),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0,0,255),
    #     3
    # )

    cv2.imshow("Frame", no_nose )
    key = cv2.waitKey(1) & 0xFF
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
