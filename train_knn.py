import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("ISL_landmark.csv")

# Features and labels
X = df.drop(columns=["target"]).values
y = df["target"].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "knn_model.joblib")
joblib.dump(scaler, "knn_scaler.joblib")
print("Saved knn_model.joblib and knn_scaler.joblib")
