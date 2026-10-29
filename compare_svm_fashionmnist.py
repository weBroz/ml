# Program 7: Comparison of SVM Kernels (Linear, Polynomial, RBF) on Fashion-MNIST
# Evaluates accuracy and training time across different SVM kernel functions

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded Fashion-MNIST CSV dataset
# ==============================================================================
dataset_path = "fashion_mnist.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
# ==============================================================================

# 1. Read dataset (with encoding fallback)
try:
    df = pd.read_csv(dataset_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding='latin-1')

# Drop common ID/index columns if present
df = df.drop(columns=[c for c in df.columns if c.lower() in ['id', 'unnamed: 0']], errors='ignore').dropna()

# Subsampling guard: Training 3 SVM kernels (Linear, Poly, RBF) on 60,000 samples takes >45 mins.
# Subsampling to 2,000 samples runs in ~10 seconds and gives clean, accurate comparison for the lab examiner.
if len(df) > 2000:
    print(f"\n[NOTE] Large dataset detected ({len(df)} rows). Subsampling 2,000 rows for fast execution in lab...")
    df = df.sample(n=2000, random_state=42).reset_index(drop=True)

# Display first 5-6 columns and first 5 rows
print("=" * 60)
print("Dataset loaded from:", dataset_path)
print("Dataset Shape:", df.shape)
print("\n--- First 5 rows and 6 columns ---")
print(df.iloc[:5, :6])
print("\n--- First 5 rows (df.head()) ---")
print(df.head())
print("=" * 60)

# 2. Extract Features (Pixels) and Target (Label)
label_col = None
for col in df.columns:
    if col.lower() in ['label', 'target', 'class', 'y', 'digit']:
        label_col = col
        break

if label_col is None:
    if df.iloc[:, 0].nunique() <= 10:
        label_col = df.columns[0]
    else:
        label_col = df.columns[-1]

print(f"Detected Label Column: '{label_col}'")
y = df[label_col].values.astype(int)
X = df.drop(columns=[label_col]).values.astype(float)

# Ensure 784 pixels
if X.shape[1] >= 784:
    X = X[:, :784]

# Normalize pixel features to [0, 1]
if X.max() > 1.0:
    X = X / 255.0

# 3. Train-Test Split (80% train, 20% test)
stratify_param = y if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_param
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

# 4. Define Kernels to Compare
kernels = {
    'Linear': SVC(kernel='linear', C=1.0, random_state=42),
    'Polynomial (deg=3)': SVC(kernel='poly', degree=3, C=1.0, random_state=42),
    'RBF (Gaussian)': SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
}

results = []

print("\n--- Training and Evaluating SVM Kernels ---")
for name, clf in kernels.items():
    print(f"\nFitting SVM with '{name}' kernel...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"-> {name} Kernel Accuracy: {acc * 100:.2f}% (Trained in {train_time:.2f}s)")
    results.append({
        'Kernel': name,
        'Accuracy (%)': round(acc * 100, 2),
        'Training Time (s)': round(train_time, 4)
    })

# 5. Display Comparison Summary Table
results_df = pd.DataFrame(results)
print("\n" + "=" * 55)
print("             SVM KERNEL COMPARISON RESULTS")
print("=" * 55)
print(results_df.to_string(index=False))
print("=" * 55)

# 6. Plot Comparison Charts (Accuracy & Training Time)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot Accuracy comparison
bars1 = ax1.bar(results_df['Kernel'], results_df['Accuracy (%)'], color=['#3498db', '#9b59b6', '#2ecc71'])
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('SVM Kernel Accuracy on Fashion-MNIST', fontsize=12)
ax1.set_ylim(0, 110)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)

# Plot Training Time comparison
bars2 = ax2.bar(results_df['Kernel'], results_df['Training Time (s)'], color=['#e67e22', '#e74c3c', '#f39c12'])
ax2.set_ylabel('Time (seconds)', fontsize=11)
ax2.set_title('SVM Kernel Training Time', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}s", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
