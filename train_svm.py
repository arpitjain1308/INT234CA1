import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("ISL_landmark.csv")  # change to your file name

# 2. Features (X) and labels (y)
X = df.drop(columns=["target"]).values   # all columns except target
y = df["target"].values                  # 0–25 for A–Z

# 3. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train SVM model
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7. Save model and scaler for real-time use
joblib.dump(model, "svm_model.joblib")
joblib.dump(scaler, "svm_scaler.joblib")
print("Saved svm_model.joblib and svm_scaler.joblib")
