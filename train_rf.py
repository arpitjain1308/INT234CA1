import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("ISL_landmark.csv")

X = df.drop(columns=["target"]).values
y = df["target"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale (RF doesn’t strictly need it, but we keep pipeline consistent)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, "rf_model.joblib")
joblib.dump(scaler, "rf_scaler.joblib")
print("Saved rf_model.joblib and rf_scaler.joblib")