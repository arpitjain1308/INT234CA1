import cv2
import mediapipe as mp
import numpy as np
import joblib
import string

# Load models + scalers
svm_model = joblib.load("svm_model.joblib")
svm_scaler = joblib.load("svm_scaler.joblib")

knn_model = joblib.load("knn_model.joblib")
knn_scaler = joblib.load("knn_scaler.joblib")

logreg_model = joblib.load("logreg_model.joblib")
logreg_scaler = joblib.load("logreg_scaler.joblib")

rf_model = joblib.load("rf_model.joblib")
rf_scaler = joblib.load("rf_scaler.joblib")

models = [
    ("SVM", svm_model, svm_scaler),
    ("KNN", knn_model, knn_scaler),
    ("LogReg", logreg_model, logreg_scaler),
    ("RandomForest", rf_model, rf_scaler)
]

# 0–25 → A–Z
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
    results = hands.process(imgRGB)

    fv = extract_features(results)

    h, w, _ = frame.shape
    half_h, half_w = h//2, w//2

    # base frame with landmarks drawn (for background)
    base = frame.copy()
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(base, lm, mp_hands.HAND_CONNECTIONS)

    panels = []
    for i, (name, model, scaler) in enumerate(models):
        sub = cv2.resize(base, (half_w, half_h))

        fv_scaled = scaler.transform(fv)
        pred_idx = model.predict(fv_scaled)[0]
        letter = label_map.get(pred_idx, "?")

        # header
        cv2.rectangle(sub, (0,0), (half_w, 40), (0,0,0), -1)
        cv2.putText(sub, name, (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # prediction text
        cv2.putText(sub, f"Pred: {letter}", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2)

        # probability (if available)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(fv_scaled)[0]
            top_idx = np.argmax(probs)
            top_letter = label_map.get(top_idx, "?")
            top_conf = probs[top_idx]
            cv2.putText(sub, f"Top: {top_letter} ({top_conf:.2f})", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        panels.append(sub)

    # grid: 2x2
    top_row = np.hstack((panels[0], panels[1]))
    bottom_row = np.hstack((panels[2], panels[3]))
    final = np.vstack((top_row, bottom_row))

    cv2.putText(final, "Press q to quit", (10, final.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Four-Model Live Comparison", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()