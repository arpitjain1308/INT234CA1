import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, multi_class="multinomial")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, "logreg_model.joblib")
joblib.dump(scaler, "logreg_scaler.joblib")
print("Saved logreg_model.joblib and logreg_scaler.joblib")