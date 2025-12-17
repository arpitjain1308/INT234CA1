import cv2
import mediapipe as mp
import os
import csv

DATASET_DIR = "ISL/data"  # change if needed
OUTPUT_FILE = "landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

header = ["label", "uses_two_hands"]
for side in ["left", "right"]:
    for i in range(21):
        header += [f"{side}_x_{i}", f"{side}_y_{i}", f"{side}_z_{i}"]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            left = [(0,0,0)] * 21
            right = [(0,0,0)] * 21
            uses_two_hands = 0

            if result.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    side = handedness.classification[0].label.lower()
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.append((lm.x, lm.y, lm.z))

                    if "right" in side:
                        right = coords
                    else:
                        left = coords

                if any(p!=(0,0,0) for p in left) and any(p!=(0,0,0) for p in right):
                    uses_two_hands = 1

            row = [label, uses_two_hands]
            for p in left:
                row += [p[0], p[1], p[2]]
            for p in right:
                row += [p[0], p[1], p[2]]

            writer.writerow(row)

print("Landmark CSV saved as", OUTPUT_FILE)
