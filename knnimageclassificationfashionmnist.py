# Program 4: K-Nearest Neighbors (KNN) Image Classification on Fashion-MNIST
# Classifies 28x28 pixel clothing articles into 10 categories

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Class names for Fashion-MNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

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

# 4. Train K-Nearest Neighbors Classifier
k = 5
print(f"\nTraining KNN Classifier with K={k}...")
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nKNN Test Accuracy (K={k}): {accuracy * 100:.2f}%")
print("\nClassification Report:")
unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
target_names = [class_names[i] for i in unique_labels] if max(unique_labels) < len(class_names) else None
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. Visualize Sample Predictions (28x28 Reshaped Images)
plt.figure(figsize=(12, 3))
num_display = min(6, len(X_test))

for i in range(num_display):
    plt.subplot(1, num_display, i + 1)
    img = X_test[i].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    
    true_label = class_names[y_test[i]] if y_test[i] < len(class_names) else str(y_test[i])
    pred_label = class_names[y_pred[i]] if y_pred[i] < len(class_names) else str(y_pred[i])
    
    color = 'green' if y_test[i] == y_pred[i] else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
    plt.axis('off')

plt.suptitle("KNN Fashion-MNIST Sample Predictions", fontsize=12)
plt.tight_layout()
plt.show()
