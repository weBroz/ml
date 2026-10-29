# Program 10: Multi-Layer Perceptron (MLP) Neural Network on Wine Dataset
# Multi-class classification of wine cultivars using chemical features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded Wine CSV dataset
# ==============================================================================
dataset_path = "wine.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
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
target_col = 'Wine_Class' if 'Wine_Class' in df.columns else ('class' if 'class' in df.columns else df.columns[-1])
X = df.drop(columns=[target_col]).values
y = df[target_col].values.astype(int)

print(f"\nTarget Column: '{target_col}' | Number of Features: {X.shape[1]}")
print("Classes present:", np.unique(y))

# 3. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
print(classification_report(y_test, y_pred, target_names=[f'Cultivar {c}' for c in np.unique(y)]))

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
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Class {c}' for c in np.unique(y)])
disp.plot(cmap='Purples', ax=ax2)
ax2.set_title('Confusion Matrix on Test Data', fontsize=12)

plt.tight_layout()
plt.show()
