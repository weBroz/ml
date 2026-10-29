# Program 3: Naive Bayes Text Classification (Spam/Ham or Topic Detection)
# Using CountVectorizer / TF-IDF and Multinomial Naive Bayes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==============================================================================
# LINE TO REPLACE: Provide the path to your downloaded Text/Spam CSV dataset
# ==============================================================================
dataset_path = "spam_text.csv"  # <-- REPLACE THIS LINE WITH YOUR ACTUAL CSV PATH
# ==============================================================================

# 1. Read dataset (with encoding fallback for Kaggle spam datasets)
try:
    df = pd.read_csv(dataset_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding='latin-1')

# Drop empty / unnamed artifact columns common in Kaggle's spam.csv
df = df.drop(columns=[c for c in df.columns if 'unnamed' in c.lower()], errors='ignore').dropna()

# Display first 5-6 columns and first 5 rows
print("=" * 60)
print("Dataset loaded from:", dataset_path)
print("Dataset Shape:", df.shape)
print("\n--- First 5 rows and 6 columns ---")
print(df.iloc[:5, :6])
print("\n--- First 5 rows (df.head()) ---")
print(df.head())
print("=" * 60)

# 2. Extract Text Feature and Target Label
text_col = None
label_col = None

for col in df.columns:
    col_lower = col.lower()
    if any(k in col_lower for k in ['message', 'text', 'sms', 'email', 'body', 'content', 'v2']):
        text_col = col
    elif any(k in col_lower for k in ['category', 'label', 'class', 'target', 'type', 'v1']):
        label_col = col

# Fallback: identify text column as longest average length string
if text_col is None or label_col is None:
    lengths = {col: df[col].astype(str).str.len().mean() for col in df.columns}
    sorted_cols = sorted(lengths.items(), key=lambda x: x[1])
    label_col = sorted_cols[0][0]
    text_col = sorted_cols[-1][0]

print(f"\nUsing Text Column: '{text_col}' | Label Column: '{label_col}'")
X = df[text_col].fillna('').astype(str)
y = df[label_col].astype(str)

# 3. Train-Test Split (80% train, 20% test)
stratify_param = y if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_param
)

# 4. Build Text Classification Pipeline (Vectorize -> TF-IDF -> MultinomialNB)
text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words='english', lowercase=True)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

# 5. Train Model
text_clf.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = text_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Test on Sample Live Messages
sample_messages = [
    "Congratulations! You won a cash prize. Claim your free reward now.",
    "Hey, are we meeting for lunch at 1 PM today?",
    "URGENT: Your account credentials need verification immediately."
]
sample_preds = text_clf.predict(sample_messages)

print("\n--- Sample Predictions ---")
for msg, pred in zip(sample_messages, sample_preds):
    print(f"Message: \"{msg}\"")
    print(f"--> Predicted: [{pred}]\n")

# 8. Plot Confusion Matrix
classes = np.unique(y)
cm = confusion_matrix(y_test, y_pred, labels=classes)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix - Naive Bayes Text Classification")
plt.tight_layout()
plt.show()
