
from pathlib import Path

from modules.ml_comparison import (
    prepare_dataset,
    train_and_compare_models
)


# Dataset location
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "data" / "raw" / "customer_feedback.csv"


# Prepare dataset
(
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test,
    tfidf
) = prepare_dataset(DATASET_PATH)


print("\nDataset preparation completed.")

print("Training records:", X_train_tfidf.shape[0])
print("Testing records:", X_test_tfidf.shape[0])


# Train and compare algorithms
results, matrices, trained_models = train_and_compare_models(
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test
)


# Display comparison
print("\nMODEL COMPARISON RESULTS")

print(results.to_string(index=False))


# Display confusion matrices
print("\nCONFUSION MATRICES")

for model_name, matrix in matrices.items():

    print(f"\n{model_name}")

    print(matrix)


# Best model
best_model = results.iloc[0]

print("\nBEST PERFORMING MODEL")

print(best_model)
