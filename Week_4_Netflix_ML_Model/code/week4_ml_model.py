# ============================================================
# WEEK 4 - MACHINE LEARNING MODEL DEVELOPMENT AND EVALUATION
# Dataset: Netflix Movies and TV Shows
# Model: Logistic Regression
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../data", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "../data/netflix_titles.csv"

df = pd.read_csv(file_path, encoding="latin1")

print("\nDataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 3. INITIAL DATA INSPECTION
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ============================================================
# 5. DATA CLEANING
# ============================================================

# Convert release year to numeric
df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)

# Convert date_added to datetime
df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)

# Extract useful date information
df["added_year"] = df["date_added"].dt.year

# Fill missing categorical values
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Unknown")
df["duration"] = df["duration"].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("Unknown")

# Fill missing numeric values
df["release_year"] = df["release_year"].fillna(
    df["release_year"].median()
)

df["added_year"] = df["added_year"].fillna(
    df["added_year"].median()
)


# ============================================================
# 6. TARGET ENCODING
# ============================================================

# Movie = 0
# TV Show = 1

df["target"] = df["type"].map({
    "Movie": 0,
    "TV Show": 1
})

# Remove records where target could not be created
df = df.dropna(subset=["target"])

df["target"] = df["target"].astype(int)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

# Number of countries associated with title
df["country_count"] = df["country"].apply(
    lambda x: len(str(x).split(","))
)

# Number of genres
df["genre_count"] = df["listed_in"].apply(
    lambda x: len(str(x).split(","))
)

# Number of cast members
df["cast_count"] = df["cast"].fillna("").apply(
    lambda x: len(str(x).split(","))
)


# ============================================================
# 8. SELECT FEATURES
# ============================================================

features = [
    "release_year",
    "added_year",
    "rating",
    "country_count",
    "genre_count",
    "cast_count"
]

X = df[features]
y = df["target"]


print("\nSelected Features:")
print(features)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = [
    "release_year",
    "added_year",
    "country_count",
    "genre_count",
    "cast_count"
]

categorical_features = [
    "rating"
]


# ============================================================
# 11. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 12. CREATE MACHINE LEARNING MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# 13. CREATE COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 15. MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 16. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Movie",
            "TV Show"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Movie",
        "TV Show"
    ],
    yticklabels=[
        "Movie",
        "TV Show"
    ]
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "../visualizations/01_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 19. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Logistic Regression"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "../visualizations/02_roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# 20. PERFORMANCE COMPARISON
# ============================================================

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": auc
}

plt.figure(figsize=(9, 5))

bars = plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(0, 1)

plt.title(
    "Logistic Regression Performance Metrics"
)

plt.ylabel("Score")

for bar, value in zip(
    bars,
    metrics.values()
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "../visualizations/03_model_performance.png",
    dpi=300
)

plt.close()


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})

results.to_csv(
    "../data/model_results.csv",
    index=False
)


# ============================================================
# 22. SAVE PREDICTIONS
# ============================================================

prediction_results = X_test.copy()

prediction_results["Actual"] = y_test.values
prediction_results["Predicted"] = y_pred
prediction_results["Probability_TV_Show"] = y_probability

prediction_results.to_csv(
    "../data/predictions.csv",
    index=False
)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("WEEK 4 ML PROJECT COMPLETED")
print("========================================")

print("\nGenerated files:")

print("1. ../visualizations/01_confusion_matrix.png")
print("2. ../visualizations/02_roc_curve.png")
print("3. ../visualizations/03_model_performance.png")
print("4. ../data/model_results.csv")
print("5. ../data/predictions.csv")

print("\nAll tasks completed successfully!")