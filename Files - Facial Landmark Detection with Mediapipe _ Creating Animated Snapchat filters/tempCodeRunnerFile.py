        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            if left_eye_status[face_num] == 'OPEN':
                frame = detector.masking(frame, left_eye, face_landmarks,
                                'LEFT EYE', detector.mpFaceMesh.FACEMESH_LEFT_EYE)
            if right_eye_status[face_num] == 'OPEN': 
                frame = detector.masking(frame, right_eye, face_landmarks,
                                'RIGHT EYE', detector.mpFaceMesh.FACEMESH_RIGHT_EYE)
            if mouth_status[face_num] == 'OPEN':
                frame = detector.masking(frame, smoke_frame, face_landmarks, 
                                'MOUTH', detector.mpFaceMesh.FACEMESH_LIPS)