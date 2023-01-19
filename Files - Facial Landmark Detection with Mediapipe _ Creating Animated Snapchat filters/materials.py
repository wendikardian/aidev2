import cv2
import mediapipe

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode = True)

# img = cv2.imread('media/sample6.jpg')
cap = cv2.VideoCapture(0)

# results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	
# landmarks = results.multi_face_landmarks[0]

facial_areas = {
    'Contours': mp_face_mesh.FACEMESH_CONTOURS
    , 'Lips': mp_face_mesh.FACEMESH_LIPS
    , 'Face_oval': mp_face_mesh.FACEMESH_FACE_OVAL
    , 'Left_eye': mp_face_mesh.FACEMESH_LEFT_EYE
    , 'Left_eye_brow': mp_face_mesh.FACEMESH_LEFT_EYEBROW
    , 'Right_eye': mp_face_mesh.FACEMESH_RIGHT_EYE
    , 'Right_eye_brow': mp_face_mesh.FACEMESH_RIGHT_EYEBROW
    , 'Tesselation': mp_face_mesh.FACEMESH_TESSELATION
}

def plot_landmark(frame, facial_area_obj):    
    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
 
        relative_source = (int(frame.shape[1] * source.x), int(frame.shape[0] * source.y))
        relative_target = (int(frame.shape[1] * target.x), int(frame.shape[0] * target.y))
 
        cv2.line(frame, relative_source, relative_target, (255, 255, 255), thickness = 2)


while True:
    ret, frame = cap.read()
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	
    landmarks = results.multi_face_landmarks[0]
    plot_landmark(frame, mp_face_mesh.FACEMESH_CONTOURS)
    cv2.imshow('frame', frame)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
      
cv2.destroyAllWindows()  