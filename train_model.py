"""
train_model.py
Task 1: Data Understanding and Preprocessing
Task 2: Model Development

Trains a classification model on the Heart Disease dataset and
saves the trained model + scaler using Joblib.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Task 1: Data Understanding and Preprocessing
# -----------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head())

# 3. Identify numerical features and target variable
target_col = "target"
numerical_features = [c for c in df.columns if c != target_col]
print("\nNumerical features:", numerical_features)
print("Target variable:", target_col)

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# -----------------------------
# Task 2: Model Development
# -----------------------------

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and scaler using Joblib
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(numerical_features, "feature_names.pkl")

print("\nModel, scaler, and feature list saved successfully.")
