import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
#tambahkan code di bawah ini
mpDraw = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=4) 

while True:

    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image=frame,
                                  landmark_list=face_landmarks, connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_tesselation_style()
                                  )

            mpDraw.draw_landmarks(image=frame,
                                  landmark_list=face_landmarks,
                                  connections=mpFaceMesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_contours_style())

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
