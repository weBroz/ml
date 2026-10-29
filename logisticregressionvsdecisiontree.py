# Program 6: Performance Comparison: Logistic Regression vs Decision Tree Classifier
# Compares classification metrics (Accuracy, Precision, Recall, F1-Score) on a dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded Classification CSV dataset
# ==============================================================================
dataset_path = "heart_disease.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
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
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col].values

# Convert any categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# 3. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling (essential for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Model 1: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# 6. Train Model 2: Decision Tree Classifier
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# 7. Evaluate and Compare Metrics
metrics_dict = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Logistic Regression': [
        round(accuracy_score(y_test, y_pred_lr), 4),
        round(precision_score(y_test, y_pred_lr, average='weighted'), 4),
        round(recall_score(y_test, y_pred_lr, average='weighted'), 4),
        round(f1_score(y_test, y_pred_lr, average='weighted'), 4)
    ],
    'Decision Tree': [
        round(accuracy_score(y_test, y_pred_dt), 4),
        round(precision_score(y_test, y_pred_dt, average='weighted'), 4),
        round(recall_score(y_test, y_pred_dt, average='weighted'), 4),
        round(f1_score(y_test, y_pred_dt, average='weighted'), 4)
    ]
}

comparison_df = pd.DataFrame(metrics_dict)
print("\n" + "=" * 50)
print("       MODEL PERFORMANCE COMPARISON")
print("=" * 50)
print(comparison_df.to_string(index=False))
print("=" * 50)

print("\n--- Logistic Regression Classification Report ---")
print(classification_report(y_test, y_pred_lr))

print("--- Decision Tree Classification Report ---")
print(classification_report(y_test, y_pred_dt))

# 8. Plot Comparison Bar Chart
x_indices = np.arange(len(metrics_dict['Metric']))
bar_width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x_indices - bar_width/2, comparison_df['Logistic Regression'], width=bar_width, label='Logistic Regression', color='#1f77b4')
plt.bar(x_indices + bar_width/2, comparison_df['Decision Tree'], width=bar_width, label='Decision Tree', color='#ff7f0e')

plt.xlabel('Evaluation Metrics', fontsize=11)
plt.ylabel('Score (0.0 to 1.0)', fontsize=11)
plt.title('Comparison: Logistic Regression vs Decision Tree', fontsize=13)
plt.xticks(x_indices, metrics_dict['Metric'], fontsize=10)
plt.ylim(0, 1.15)
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values above bars
for i in x_indices:
    plt.text(i - bar_width/2, comparison_df['Logistic Regression'][i] + 0.02, f"{comparison_df['Logistic Regression'][i]:.2f}", ha='center', fontsize=9)
    plt.text(i + bar_width/2, comparison_df['Decision Tree'][i] + 0.02, f"{comparison_df['Decision Tree'][i]:.2f}", ha='center', fontsize=9)

plt.tight_layout()
plt.show()
