# Program 5: Customer Segmentation using Decision Tree Classifier
# Classifies customer segments based on demographic and spending features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded Customer Segmentation CSV
# ==============================================================================
dataset_path = "customer_segmentation.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
# ==============================================================================

# 1. Read dataset (with encoding fallback)
try:
    df = pd.read_csv(dataset_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding='latin-1')

# Drop common ID/index columns if present (e.g. CustomerID)
df = df.drop(columns=[c for c in df.columns if c.lower() in ['customerid', 'customer_id', 'id', 'unnamed: 0']], errors='ignore').dropna()

# Display first 5-6 columns and first 5 rows
print("=" * 60)
print("Dataset loaded from:", dataset_path)
print("Dataset Shape:", df.shape)
print("\n--- First 5 rows and 6 columns ---")
print(df.iloc[:5, :6])
print("\n--- First 5 rows (df.head()) ---")
print(df.head())
print("=" * 60)

# 2. Robust Target and Feature Extraction
target_col = None
for col in df.columns:
    if col.lower() in ['customer_segment', 'segment', 'cluster', 'target', 'class', 'category']:
        target_col = col
        break

# If raw Mall_Customers.csv is used (which has no target column), auto-generate segments from Spending Score
if target_col is None:
    spending_cols = [c for c in df.columns if any(k in c.lower() for k in ['spend', 'score'])]
    if spending_cols:
        target_col = 'Customer_Segment'
        df[target_col] = pd.qcut(df[spending_cols[0]], q=3, labels=['Low Spender', 'Medium Spender', 'High Spender'])
        print(f"Auto-generated segment target '{target_col}' based on '{spending_cols[0]}'.")
    else:
        target_col = df.columns[-1]

# Separate features
feature_df = df.drop(columns=[target_col])

# Encode all non-numeric columns (e.g. Gender, Genre) using LabelEncoder
encoded_features = feature_df.copy()
for col in encoded_features.columns:
    if not pd.api.types.is_numeric_dtype(encoded_features[col]):
        le = LabelEncoder()
        encoded_features[col] = le.fit_transform(encoded_features[col].astype(str))

X = encoded_features.astype(float)
y = df[target_col].astype(str)

print(f"\nTarget Variable: '{target_col}'")
print("Target Class Distribution:")
print(y.value_counts())
print("\nFeatures used for training:", list(X.columns))

# 3. Train-Test Split (80% train, 20% test)
stratify_param = y if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_param
)

# 4. Train Decision Tree Classifier
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=5,
    random_state=42
)
clf.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nDecision Tree Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Feature Importance Analysis
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# 7. Visualize the Decision Tree Structure
plt.figure(figsize=(16, 9))
plot_tree(
    clf,
    feature_names=list(X.columns),
    class_names=[str(c) for c in clf.classes_],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Customer Segmentation Decision Tree Structure", fontsize=14)
plt.tight_layout()
plt.show()
