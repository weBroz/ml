# Program 9: Neural Network (MLP) on MNIST Dataset: Comparing Activation Functions
# Evaluates Identity, Logistic (Sigmoid), Tanh, and ReLU activation functions

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded MNIST CSV dataset
# ==============================================================================
dataset_path = "mnist.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
# ==============================================================================

# 1. Read dataset (with encoding fallback)
try:
    df = pd.read_csv(dataset_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding='latin-1')

# Drop common ID/index columns if present
df = df.drop(columns=[c for c in df.columns if c.lower() in ['id', 'unnamed: 0']], errors='ignore').dropna()

# Subsampling guard: Training 4 separate Neural Networks (150 epochs each) on 60,000 rows takes 25+ minutes.
# Subsampling to 3,000 rows converges smoothly in ~10-15 seconds and generates complete loss curves.
if len(df) > 3000:
    print(f"\n[NOTE] Large dataset detected ({len(df)} rows). Subsampling 3,000 rows for fast execution in lab...")
    df = df.sample(n=3000, random_state=42).reset_index(drop=True)

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

# 4. Activation Functions to Compare
activations = ['identity', 'logistic', 'tanh', 'relu']
activation_labels = {
    'identity': 'Identity (Linear)',
    'logistic': 'Logistic (Sigmoid)',
    'tanh': 'Tanh (Hyperbolic Tangent)',
    'relu': 'ReLU (Rectified Linear Unit)'
}

results = []
plt.figure(figsize=(10, 6))

print("\n--- Training MLP with Different Activation Functions ---")
for act in activations:
    print(f"\nTraining MLP with '{activation_labels[act]}' activation...")
    start_time = time.time()
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation=act,
        solver='adam',
        max_iter=150,
        random_state=42,
        early_stopping=False
    )
    mlp.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"-> {activation_labels[act]} Accuracy: {acc * 100:.2f}% (Trained in {elapsed_time:.2f}s, Iterations: {mlp.n_iter_})")
    
    results.append({
        'Activation Function': activation_labels[act],
        'Test Accuracy (%)': round(acc * 100, 2),
        'Iterations': mlp.n_iter_,
        'Training Time (s)': round(elapsed_time, 2)
    })
    
    # Plot loss curve
    plt.plot(mlp.loss_curve_, label=f"{activation_labels[act]} (Acc: {acc*100:.1f}%)", linewidth=2)

# 5. Display Summary Results Table
results_df = pd.DataFrame(results)
print("\n" + "=" * 65)
print("             ACTIVATION FUNCTION COMPARISON SUMMARY")
print("=" * 65)
print(results_df.to_string(index=False))
print("=" * 65)

# 6. Customize Loss Curve Plot
plt.xlabel("Epochs / Iterations", fontsize=11)
plt.ylabel("Training Loss", fontsize=11)
plt.title("MLP Training Loss Convergence across Activation Functions (MNIST)", fontsize=13)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
