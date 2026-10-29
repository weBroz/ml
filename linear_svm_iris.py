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

# 1. Read dataset (with encoding fallback)
try:
    df = pd.read_csv(dataset_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding='latin-1')

# Drop common ID/index columns if present (e.g. from Kaggle)
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

# 2. Robust Column Extraction (handles Kaggle, UCI, GitHub column names)
def find_column(keywords):
    for col in df.columns:
        norm = col.lower().replace('_', '').replace(' ', '').replace('.', '')
        if all(k in norm for k in keywords):
            return col
    return None

sl_col = find_column(['sepal', 'length'])
pl_col = find_column(['petal', 'length'])

if sl_col and pl_col:
    X = df[[sl_col, pl_col]].values.astype(float)
    print(f"Using features: '{sl_col}' and '{pl_col}' for 2D visualization.")
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_indices = [0, 2] if len(numeric_cols) > 2 else [0, 1]
    X = df[numeric_cols[feature_indices]].values.astype(float)
    print(f"Using numeric features: {list(numeric_cols[feature_indices])}")

# Detect target column (e.g. Species, Variety, class, target)
target_col = df.columns[-1]
for col in df.columns:
    if col.lower() in ['species', 'variety', 'class', 'target', 'label']:
        target_col = col
        break

print(f"Using target column: '{target_col}'")
# Setosa = 1, others = 0
y = (df[target_col].astype(str).str.contains('setosa', case=False) | (df[target_col].astype(str) == '0')).astype(int).values

# 3. Train-Test Split (80% train, 20% test)
stratify_param = y if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_param
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
print(classification_report(y_test, y_pred, target_names=['Non-Setosa (0)', 'Setosa (1)'], zero_division=0))

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

plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Linear SVM Decision Boundary: Setosa vs Non-Setosa')
plt.legend()
plt.tight_layout()
plt.show()
