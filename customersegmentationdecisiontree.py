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
target_candidates = ['Customer_Segment', 'Segment', 'Cluster', 'target', 'Target']
target_col = next((c for c in target_candidates if c in df.columns), df.columns[-1])

# Drop identifier columns like CustomerID
drop_cols = [c for c in df.columns if 'id' in c.lower()] + [target_col]
feature_df = df.drop(columns=drop_cols)

# Encode any categorical columns (e.g. Gender)
encoded_features = feature_df.copy()
for col in encoded_features.columns:
    if encoded_features[col].dtype == 'object' or encoded_features[col].dtype.name == 'category':
        le = LabelEncoder()
        encoded_features[col] = le.fit_transform(encoded_features[col].astype(str))

X = encoded_features
y = df[target_col].astype(str)

print(f"\nTarget Variable: '{target_col}'")
print("Target Class Distribution:")
print(y.value_counts())
print("\nFeatures used for training:", list(X.columns))

# 3. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
print(classification_report(y_test, y_pred))

# 6. Feature Importance Analysis
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# 7. Visualize the Decision Tree Structure
plt.figure(figsize=(16, 9))
plot_tree(
    clf,
    feature_names=list(X.columns),
    class_names=sorted(y.unique()),
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Customer Segmentation Decision Tree Structure", fontsize=14)
plt.tight_layout()
plt.show()
