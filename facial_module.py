import cv2
import numpy as np
import mediapipe as mp
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Lightweight ONNX emotion model — no TensorFlow involved
emotion_model = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, eye_idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    vert1 = np.linalg.norm(pts[1] - pts[5])
    vert2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return (vert1 + vert2) / (2.0 * horiz + 1e-6)


def head_pose_stability(landmarks, w, h):
    nose = landmarks[1]
    dx = abs(nose.x * w - w / 2) / (w / 2)
    dy = abs(nose.y * h - h / 2) / (h / 2)
    return 1 - min(1.0, (dx + dy) / 2)


def analyze_frame(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"face_detected": False}

    landmarks = results.multi_face_landmarks[0].landmark
    ear = (eye_aspect_ratio(landmarks, LEFT_EYE, w, h) +
           eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)) / 2
    pose_stability = head_pose_stability(landmarks, w, h)

    try:
        _, scores = emotion_model.predict_emotions(rgb, logits=False)
        emotions = {
            emotion_model.idx_to_class_labels[i]: float(scores[i]) * 100
            for i in range(len(scores))
        }
    except Exception:
        emotions = {}

    return {
        "face_detected": True,
        "eye_aspect_ratio": ear,
        "head_pose_stability": pose_stability,
        "emotions": emotions,
    }


def label_from_score(score: float) -> str:
    if score >= 60:
        return "energetic"
    elif score >= 45:
        return "confident"
    elif score >= 30:
        return "nervous"
    else:
        return "low_confidence"


def composite_score(analysis, blink_rate_per_min):
    if not analysis.get("face_detected"):
        return {"label": "no_face", "score": 0}

    emo = analysis["emotions"]
    happy = emo.get("Happiness", 0)
    neutral = emo.get("Neutral", 0)
    fear = emo.get("Fear", 0)
    sad = emo.get("Sadness", 0)

    stability = analysis["head_pose_stability"]
    ear = analysis["eye_aspect_ratio"]
    blink_penalty = min(1.0, blink_rate_per_min / 40)

    confidence_raw = (
        0.35 * (happy + neutral) / 100 +
        0.30 * stability +
        0.20 * (1 - blink_penalty) +
        0.15 * min(1.0, ear / 0.3)
    )
    nervous_raw = 0.5 * (fear + sad) / 100 + 0.3 * blink_penalty + 0.2 * (1 - stability)

    score = round(max(0, min(100, confidence_raw * 100)), 1)
    label = label_from_score(score)

    return {"label": label, "score": score, "nervous_index": round(nervous_raw * 100, 1)}