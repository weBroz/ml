# Program 1: Linear SVM Classification on Iris Dataset (Setosa vs Non-Setosa)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded Iris CSV dataset
# ==============================================================================
dataset_path = "iris.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
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

# 2. Extract Features and Target
# Using Sepal Length and Petal Length for 2D decision boundary visualization
feature_cols = [c for c in df.columns if any(k in c.lower() for k in ['sepallength', 'petallength'])]
if len(feature_cols) >= 2:
    X = df[feature_cols[:2]].values
else:
    # Fallback to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_indices = [0, 2] if len(numeric_cols) > 2 else [0, 1]
    X = df[numeric_cols[feature_indices]].values

# Target: Setosa = 1, Non-Setosa = 0
target_col = df.columns[-1]
y = (df[target_col].astype(str).str.contains('setosa', case=False) | (df[target_col] == 0)).astype(int).values

# 3. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create Linear SVM pipeline with feature scaling
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=1.0)
)

# 5. Train Model
model.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {round(accuracy_score(y_test, y_pred), 4)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Setosa (0)', 'Setosa (1)']))

# 7. Create grid for decision boundary visualization
scaler = model.named_steps['standardscaler']
svm = model.named_steps['svc']

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)

Z = svm.predict(grid_scaled).reshape(xx.shape)
decision = svm.decision_function(grid_scaled).reshape(xx.shape)

# 8. Plot decision boundary and margins
plt.figure(figsize=(9, 6))

plt.contourf(xx, yy, Z, alpha=0.15, cmap='coolwarm')
plt.contour(
    xx, yy, decision,
    levels=[-1, 0, 1],
    colors=['blue', 'black', 'blue'],
    linestyles=['--', '-', '--']
)

# Plot training points
plt.scatter(
    X_train[y_train == 1, 0], X_train[y_train == 1, 1],
    label='Setosa (1)', marker='o', edgecolors='k'
)
plt.scatter(
    X_train[y_train == 0, 0], X_train[y_train == 0, 1],
    label='Non-Setosa (0)', marker='s', edgecolors='k'
)

# Plot support vectors
support_vectors_original = scaler.inverse_transform(svm.support_vectors_)
plt.scatter(
    support_vectors_original[:, 0],
    support_vectors_original[:, 1],
    s=130, facecolors='none', edgecolors='green',
    linewidths=1.5, label='Support Vectors'
)

plt.xlabel('Feature 1 (Sepal Length)')
plt.ylabel('Feature 2 (Petal Length)')
plt.title('Linear SVM Decision Boundary: Setosa vs Non-Setosa')
plt.legend()
plt.tight_layout()
plt.show()
