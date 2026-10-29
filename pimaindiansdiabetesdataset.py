# Program 2: Classification on Pima Indians Diabetes Dataset
# Predict whether a patient has diabetes (Outcome: 0 or 1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded Diabetes CSV dataset
# ==============================================================================
dataset_path = "diabetes.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
# ==============================================================================

# 1. Read dataset (with encoding fallback)
try:
    df = pd.read_csv(dataset_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding='latin-1')

# Drop common ID/index columns if present
df = df.drop(columns=[c for c in df.columns if c.lower() in ['id', 'unnamed: 0']], errors='ignore').dropna()

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
target_candidates = ['outcome', 'diabetes', 'class', 'target', 'label']
target_col = df.columns[-1]
for col in df.columns:
    if col.lower() in target_candidates:
        target_col = col
        break

print(f"Using target column: '{target_col}'")
X = df.drop(columns=[target_col])

# Safely encode target using LabelEncoder to handle text, numbers, or boolean
le = LabelEncoder()
y = le.fit_transform(df[target_col].astype(str))

# Convert categorical features to numeric if any
X = pd.get_dummies(X, drop_first=True)

# 3. Train-Test Split (80% train, 20% test)
stratify_param = y if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_param
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
display_labels = [f"Class {c}" for c in le.classes_] if len(le.classes_) == 2 else None
print(classification_report(y_test, y_pred, target_names=display_labels, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 7. Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix - Diabetes Classification")
plt.tight_layout()
plt.show()
