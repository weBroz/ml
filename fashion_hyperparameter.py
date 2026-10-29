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

# 2. Extract Features (Pixels) and Target (Label)
label_col = 'label' if 'label' in df.columns else df.columns[0]
y = df[label_col].values.astype(int)
X = df.drop(columns=[label_col]).values.astype(float)

# Normalize pixel features to [0, 1]
X = X / 255.0

# 3. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
print(classification_report(y_test, y_pred))

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
