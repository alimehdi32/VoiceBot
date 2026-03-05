import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# # ==========================
# # 1 Load Dataset
# # ==========================

print("\nLoading dataset...")

df = pd.read_csv("datasets/intent_dataset.csv")

print("Dataset Loaded Successfully")

print("\nDataset Shape:", df.shape)


# # ==========================
# # 2 Features and Labels
# # ==========================

X = df["text"]

y = df["intent"]


# # ==========================
# # 3 Train Test Split
# # ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

print("\nTrain Samples:", len(X_train))

print("Test Samples:", len(X_test))


# # ==========================
# # 4 TF-IDF Vectorization
# # ==========================

print("\nTraining TF-IDF Vectorizer...")

vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1,2)

)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

print("Vectorization Complete")


# # ==========================
# # 5 Train Logistic Regression
# # ==========================

print("\nTraining Logistic Regression Model...")

model = LogisticRegression(
    C=100,
    l1_ratio=0.5,
    max_iter=200,
    penalty="elasticnet",
    solver="saga"
)

model.fit(

    X_train_vec,

    y_train

)

print("Model Training Completed")


# # ==========================
# # 6 Predictions
# # ==========================

predictions = model.predict(

    X_test_vec

)


# # ==========================
# # 7 Evaluation Metrics
# # ==========================

accuracy = accuracy_score(

    y_test,

    predictions

)

precision = precision_score(

    y_test,

    predictions,

    average="weighted"

)

recall = recall_score(

    y_test,

    predictions,

    average="weighted"

)

f1 = f1_score(

    y_test,

    predictions,

    average="weighted"

)


print("\nEvaluation Metrics")

print("------------------")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)


print("\nClassification Report:\n")

print(

    classification_report(

        y_test,

        predictions

    )

)


# # ==========================
# # 8 Confusion Matrix
# # ==========================

print("\nGenerating Confusion Matrix...")

cm = confusion_matrix(

    y_test,

    predictions

)

plt.figure(

    figsize=(10,8)

)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=model.classes_,

    yticklabels=model.classes_

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Intent Classification Confusion Matrix")

plt.savefig(

    "datasets/confusion_matrix.png"

)

print("Confusion Matrix Saved")


# ==========================
# 9 Save Model
# ==========================

print("\nSaving Model...")

os.makedirs(

    "app/models",

    exist_ok=True

)

joblib.dump(

    model,

    "app/models/intent_model.pkl"

)

joblib.dump(

    vectorizer,

    "app/models/vectorizer.pkl"

)

print("Model Saved Successfully")

