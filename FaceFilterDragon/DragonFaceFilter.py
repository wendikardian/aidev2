import cv2
import FaceMeshDetector as fmd
detector = fmd.FaceMesh()

# cap = cv2.VideoCapture("assets/vid1.mp4")
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,960)
left_eye = cv2.imread('assets/eye1.png')
right_eye = cv2.imread('assets/eye2.png')
# smoke_animation = cv2.VideoCapture('assets/smoke_animation.mp4')
smoke_animation = cv2.VideoCapture('assets/smoke_animation.mp4')
smoke_frame_counter = 0


while True:
    ret, frame = cap.read()
    ret, smoke_frame = smoke_animation.read()
    smoke_frame_counter += 1
    if smoke_frame_counter == smoke_animation.get(cv2.CAP_PROP_FRAME_COUNT):     
        smoke_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
        smoke_frame_counter = 0
    
    frame = cv2.flip(frame, 1)
    frame_face_mesh, face_mesh_results = detector.detectFacialLandmarks(frame, detector.faceMeshVideos)
    if face_mesh_results.multi_face_landmarks:
        mouth_frame, mouth_status = detector.isOpen(frame, face_mesh_results, 'MOUTH', 
                                     threshold=15)
        left_eye_frame, left_eye_status = detector.isOpen(frame, face_mesh_results, 'LEFT EYE',  
                                        threshold=4.5)
        right_eye_frame, right_eye_status = detector.isOpen(frame, face_mesh_results, 'RIGHT EYE', 
                                         threshold=4.5)
        # Tambahkan code di bawah ini
        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            if left_eye_status[face_num] == 'OPEN':
                frame = detector.masking(frame, left_eye, face_landmarks,
                                'LEFT EYE', detector.mpFaceMesh .FACEMESH_LEFT_EYE)
            if right_eye_status[face_num] == 'OPEN': 
                frame = detector.masking(frame, right_eye, face_landmarks,
                                'RIGHT EYE', detector.mpFaceMesh.FACEMESH_RIGHT_EYE)
            if mouth_status[face_num] == 'OPEN':
                frame = detector.masking(frame, smoke_frame, face_landmarks, 
                                'MOUTH', detector.mpFaceMesh.FACEMESH_LIPS)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()

