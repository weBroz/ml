# Program 2: Classification on Pima Indians Diabetes Dataset
# Predict whether a patient has diabetes (Outcome: 0 or 1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded Diabetes CSV dataset
# ==============================================================================
dataset_path = "diabetes.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
# ==============================================================================

# 1. Read dataset
df = pd.read_csv(dataset_path)

# Display first 5-6 columns and first 5 rows
print("=" * 60)
print("Dataset loaded from:", dataset_path)
print("Dataset Shape:", df.shape)
print("\n--- First 5 rows and 6 columns ---")
print(df.iloc[:5, :6])
print("\n--- First 5 rows (df.head()) ---")
print(df.head())
print("=" * 60)

# 2. Extract Features (X) and Target (y)
target_col = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col].values

# 3. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Logistic Regression Classifier
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes (0)', 'Diabetes (1)']))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 7. Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix - Pima Indians Diabetes Classification")
plt.tight_layout()
plt.show()
