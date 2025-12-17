import cv2
import mediapipe as mp
import numpy as np
import joblib
import string

model = joblib.load("knn_model.joblib")
scaler = joblib.load("knn_scaler.joblib")

label_map = {i: ch for i, ch in enumerate(string.ascii_uppercase)}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

POINTS = 21

def extract_features(results):
    left = [(0,0,0)]*POINTS
    right = [(0,0,0)]*POINTS
    uses_two = 0.0

    if results.multi_hand_landmarks:
        for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            side = hd.classification[0].label.lower()
            pts = [(p.x,p.y,p.z) for p in lm.landmark]
            if "right" in side:
                right = pts
            else:
                left = pts

        if any(p!=(0,0,0) for p in left) and any(p!=(0,0,0) for p in right):
            uses_two = 1.0

    fv = [uses_two]
    for p in left:
        fv.extend(p)
    for p in right:
        fv.extend(p)
    
    return np.array(fv, dtype=np.float32).reshape(1,-1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(imgRGB)

    fv = extract_features(res)
    fv_scaled = scaler.transform(fv)

    pred = model.predict(fv_scaled)[0]
    pred_letter = label_map.get(pred, "?")

    cv2.putText(frame, f"KNN Prediction: {pred_letter}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("KNN Live Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
