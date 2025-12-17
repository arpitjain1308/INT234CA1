import cv2
import mediapipe as mp
import numpy as np
import joblib
import string

# Load model + scaler
model = joblib.load("svm_model.joblib")
scaler = joblib.load("svm_scaler.joblib")

# Map 0-25 → A-Z
label_map = {i: ch for i, ch in enumerate(string.ascii_uppercase)}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

POINTS = 21

def extract_features(results):
    left = [(0,0,0)] * POINTS
    right = [(0,0,0)] * POINTS
    uses_two = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            side = handedness.classification[0].label.lower()
            pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            if "right" in side:
                right = pts
            else:
                left = pts

        if any(p != (0,0,0) for p in left) and any(p != (0,0,0) for p in right):
            uses_two = 1.0

    fv = [uses_two]
    for p in left:
        fv.extend([p[0], p[1], p[2]])
    for p in right:
        fv.extend([p[0], p[1], p[2]])

    return np.array(fv, dtype=np.float32).reshape(1, -1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    fv = extract_features(results)
    fv_scaled = scaler.transform(fv)

    pred_idx = model.predict(fv_scaled)[0]
    pred_letter = label_map.get(pred_idx, "?")

    cv2.putText(frame, f"Prediction: {pred_letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Live SVM Test (A-Z)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
