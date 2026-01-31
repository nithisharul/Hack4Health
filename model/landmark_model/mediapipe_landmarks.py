import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
UPPER_LIP_IDX = [13, 14]
LOWER_LIP_IDX = [17, 18]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LEFT_EYEBROW = [105, 66]
RIGHT_EYEBROW = [334, 296]

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(eye_landmarks):
    A = euclidean(eye_landmarks[1], eye_landmarks[5])
    B = euclidean(eye_landmarks[2], eye_landmarks[4])
    C = euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_landmarks(frame):
    landmarks_data = {}
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        rgb_frame = frame[:, :, ::-1]
        result = face_mesh.process(rgb_frame)
        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            landmarks_data['landmarks'] = face.landmark
            h, w, _ = frame.shape
            pts = [[lm.x * w, lm.y * h] for lm in face.landmark]
            pts = np.array(pts)

            # Blink EAR
            left_eye_pts = pts[LEFT_EYE_IDX]
            right_eye_pts = pts[RIGHT_EYE_IDX]
            landmarks_data['blink_left'] = eye_aspect_ratio(left_eye_pts)
            landmarks_data['blink_right'] = eye_aspect_ratio(right_eye_pts)

            # Smile width/height
            left_mouth = pts[MOUTH_LEFT]
            right_mouth = pts[MOUTH_RIGHT]
            upper_lip = np.mean(pts[UPPER_LIP_IDX], axis=0)
            lower_lip = np.mean(pts[LOWER_LIP_IDX], axis=0)
            landmarks_data['smile_width'] = euclidean(left_mouth, right_mouth)
            landmarks_data['smile_height'] = euclidean(upper_lip, lower_lip)

            # Eyebrow height
            left_eyebrow = np.mean(pts[LEFT_EYEBROW], axis=0)
            right_eyebrow = np.mean(pts[RIGHT_EYEBROW], axis=0)
            landmarks_data['eyebrow_left'] = left_eyebrow[1]
            landmarks_data['eyebrow_right'] = right_eyebrow[1]
    return landmarks_data


