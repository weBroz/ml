# Program 10: Multi-Layer Perceptron (MLP) Neural Network on Wine Dataset
# Multi-class classification of wine cultivars using chemical features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded Wine CSV dataset
# ==============================================================================
dataset_path = "wine.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
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
target_col = None
for col in df.columns:
    if col.lower() in ['wine_class', 'class', 'target', 'cultivar', 'label', 'wine']:
        target_col = col
        break

if target_col is None:
    if df.iloc[:, 0].nunique() <= 5:
        target_col = df.columns[0]
    else:
        target_col = df.columns[-1]

print(f"\nTarget Column Detected: '{target_col}'")
X = df.drop(columns=[target_col]).values.astype(float)

# Robust label encoding: works on string names, 1-indexed (1,2,3), or 0-indexed (0,1,2)
le = LabelEncoder()
y = le.fit_transform(df[target_col].astype(str))

print(f"Number of Features: {X.shape[1]}")
print("Classes present:", np.unique(y))

# 3. Train-Test Split (80% train, 20% test)
stratify_param = y if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_param
)

# 4. Feature Scaling (Standardization is crucial for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build and Train Multi-Layer Perceptron (MLP) Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    random_state=42
)

print("\nTraining MLP Classifier (Architecture: Input -> 64 -> 32 -> Output)...")
mlp.fit(X_train_scaled, y_train)

# 6. Evaluate Model
y_pred = mlp.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\nMLP Test Accuracy: {acc * 100:.2f}% (Trained in {mlp.n_iter_} iterations)")
print("\nClassification Report:")
target_names = [f"Class {c}" for c in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# 7. Visualization: Loss Curve and Confusion Matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: Training Loss Curve
ax1.plot(mlp.loss_curve_, color='#8e44ad', linewidth=2)
ax1.set_xlabel('Epochs / Iterations', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title(f'MLP Loss Curve (Final Loss: {mlp.loss_:.4f})', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# Subplot 2: Confusion Matrix Display
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Purples', ax=ax2)
ax2.set_title('Confusion Matrix on Test Data', fontsize=12)

plt.tight_layout()
plt.show()
