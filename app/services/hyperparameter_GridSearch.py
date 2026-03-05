from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv("datasets/intent_dataset.csv")

X = df["text"]
y = df["intent"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# Hyperparameter Grid
param_grid = [

    # L2 penalty
    {
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "newton-cg", "sag", "saga"],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__max_iter": [100, 200, 300]
    },

    # L1 penalty
    {
        "clf__penalty": ["l1"],
        "clf__solver": ["saga"],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__max_iter": [100, 200, 300]
    },

    # ElasticNet
    {
        "clf__penalty": ["elasticnet"],
        "clf__solver": ["saga"],
        "clf__l1_ratio": [0.1, 0.5, 0.9],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__max_iter": [100, 200, 300]
    },

    # No regularization
    {
        "clf__penalty": [None],
        "clf__solver": ["lbfgs", "newton-cg", "sag", "saga"],
        "clf__max_iter": [100, 200, 300]
    }
]

# GridSearch
gridSearch = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1,
    error_score="raise"
)

print("\nStarting Grid Search for Hyperparameter Tuning...")
gridSearch.fit(X_train, y_train)
print("Grid Search Completed")

# Best Parameters
best_params = gridSearch.best_params_
print("\nBest Hyperparameters:")
print(best_params)

# Best Score
best_score = gridSearch.best_score_
print("\nBest CV Score:")
print(best_score)

# Test Accuracy
test_score = gridSearch.score(X_test, y_test)
print("\nTest Accuracy:")
print(test_score)