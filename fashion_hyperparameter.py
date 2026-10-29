# Program 8: Hyperparameter Tuning on Fashion-MNIST Dataset
# Uses GridSearchCV to find the best hyperparameters for SVM Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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

# Subsampling guard: GridSearchCV runs 36 SVM models. Running on full 60,000 samples takes hours.
# Subsampling to 1,500 samples finishes in ~15-20 seconds with clean cross-validation results.
if len(df) > 1500:
    print(f"\n[NOTE] Large dataset detected ({len(df)} rows). Subsampling 1,500 rows for fast GridSearchCV in lab...")
    df = df.sample(n=1500, random_state=42).reset_index(drop=True)

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

# 4. Define Hyperparameter Search Grid
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 0.01],
    'kernel': ['rbf', 'linear']
}

print("\n--- Hyperparameter Tuning using GridSearchCV ---")
print("Candidate parameters to search:", param_grid)

# 5. Execute GridSearchCV
base_svm = SVC(random_state=42)
grid_search = GridSearchCV(
    estimator=base_svm,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 6. Display Best Hyperparameters and Cross-Validation Score
print("\n" + "=" * 55)
print("             GRID SEARCH OPTIMIZATION RESULTS")
print("=" * 55)
print(f"Best Hyperparameters : {grid_search.best_params_}")
print(f"Best 3-Fold CV Score : {grid_search.best_score_ * 100:.2f}%")
print("=" * 55)

# 7. Evaluate the Best Estimator on Test Set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nFinal Test Accuracy with Best Model: {test_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Visualize CV Scores across parameter combinations
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results['param_combo'] = (
    "C=" + cv_results['param_C'].astype(str) + 
    ", " + cv_results['param_kernel'].astype(str) + 
    ", g=" + cv_results['param_gamma'].astype(str)
)

plt.figure(figsize=(12, 5))
plt.barh(cv_results['param_combo'], cv_results['mean_test_score'] * 100, color='#2c3e50')
plt.xlabel('Mean Cross-Validation Accuracy (%)', fontsize=11)
plt.title('Hyperparameter Combinations vs CV Accuracy (Fashion-MNIST)', fontsize=13)
plt.xlim(0, 105)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for index, value in enumerate(cv_results['mean_test_score'] * 100):
    plt.text(value + 1, index, f"{value:.1f}%", va='center', fontsize=9)

plt.tight_layout()
plt.show()
