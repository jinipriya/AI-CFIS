
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_dataset(csv_path):

    # Load dataset
    df = pd.read_csv(csv_path)

    # Combine title and content
    df["review_text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["content"].fillna("").astype(str)
    ).str.strip()

    # Remove empty reviews
    df = df[df["review_text"] != ""].copy()

    # Features and target
    X = df["review_text"]
    y = df["label"].astype(int)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    return (
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        tfidf
    )

import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_and_compare_models(
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test
):

    # Define ML algorithms
    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "SVM": LinearSVC(
            random_state=42,
            max_iter=2000
        )
    }

    results = []
    confusion_matrices = {}
    trained_models = {}

    # Train and evaluate each algorithm
    for model_name, model in models.items():

        print(f"Training {model_name}...")

        start_time = time.time()

        model.fit(X_train_tfidf, y_train)

        predictions = model.predict(X_test_tfidf)

        training_time = time.time() - start_time

        # Evaluation metrics
        accuracy = accuracy_score(
            y_test, predictions
        )

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        # Store results
        results.append({

            "Model": model_name,
            "Accuracy": round(accuracy * 100, 2),
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1 Score": round(f1 * 100, 2),
            "Training Time (seconds)": round(training_time, 2)

        })

        confusion_matrices[model_name] = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        trained_models[model_name] = model

        print(
            f"{model_name} completed. "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)

    return results_df, confusion_matrices, trained_models
