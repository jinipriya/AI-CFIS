
from pathlib import Path

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from modules.bias_analysis import (
    calculate_fairness_metrics,
    analyze_review_length
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "data" / "raw" / "customer_feedback.csv"
)

MODEL_PATH = (
    BASE_DIR / "models" / "distilbert_sentiment"
)

OUTPUT_PATH = (
    BASE_DIR / "bias_analysis_results.csv"
)

BATCH_SIZE = 16


# =====================================================
# 2. LOAD DATASET
# =====================================================

print("\nLoading customer feedback dataset...")

df = pd.read_csv(DATASET_PATH)

df["review_text"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["content"].fillna("").astype(str)
).str.strip()

df = df[df["review_text"] != ""].copy()

df.reset_index(drop=True, inplace=True)

print("Dataset loaded successfully.")
print("Total reviews:", len(df))


# =====================================================
# 3. TRAIN-TEST SPLIT
# =====================================================

X = df["review_text"]

y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset split completed.")
print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))


# =====================================================
# 4. LOAD DISTILBERT MODEL
# =====================================================

print("\nLoading saved DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH)
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH)
)

model.to("cpu")
model.eval()

print("DistilBERT model loaded successfully.")


# =====================================================
# 5. GENERATE PREDICTIONS
# =====================================================

print("\nGenerating predictions...")

predictions = []

total_batches = (
    len(X_test) + BATCH_SIZE - 1
) // BATCH_SIZE

for start in range(0, len(X_test), BATCH_SIZE):

    batch_reviews = X_test.iloc[
        start:start + BATCH_SIZE
    ].tolist()

    inputs = tokenizer(
        batch_reviews,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.inference_mode():

        outputs = model(**inputs)

        batch_predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

    predictions.extend(
        batch_predictions.tolist()
    )

    current_batch = (
        start // BATCH_SIZE
    ) + 1

    print(
        f"Processed batch {current_batch}/{total_batches}"
    )

print("\nPrediction completed successfully.")


# =====================================================
# 6. OVERALL MODEL PERFORMANCE
# =====================================================

overall_metrics = calculate_fairness_metrics(
    y_test.tolist(),
    predictions
)

print("\n======================================")
print("OVERALL MODEL PERFORMANCE")
print("======================================")

for metric, value in overall_metrics.items():

    print(f"{metric}: {value}%")


# =====================================================
# 7. REVIEW LENGTH-BASED ANALYSIS
# =====================================================

test_reviews = X_test.reset_index(
    drop=True
).to_frame(name="review_text")

y_test_reset = y_test.reset_index(drop=True)

length_metrics, detailed_results = analyze_review_length(
    test_reviews,
    y_test_reset,
    predictions
)

print("\n======================================")
print("PERFORMANCE BY REVIEW LENGTH")
print("======================================")

if not length_metrics.empty:

    print(
        length_metrics.to_string(index=False)
    )

else:

    print("No review-length groups available.")


# =====================================================
# 8. ERROR ANALYSIS
# =====================================================

detailed_results["Correct Prediction"] = (
    detailed_results["Actual"]
    == detailed_results["Predicted"]
)

detailed_results["Prediction Result"] = (
    detailed_results["Correct Prediction"]
    .map({
        True: "Correct",
        False: "Incorrect"
    })
)

correct_count = (
    detailed_results["Correct Prediction"].sum()
)

incorrect_count = (
    (~detailed_results["Correct Prediction"]).sum()
)

print("\n======================================")
print("PREDICTION ERROR SUMMARY")
print("======================================")

print("Correct predictions:", correct_count)

print("Incorrect predictions:", incorrect_count)

print(
    "Total predictions:",
    len(detailed_results)
)


# =====================================================
# 9. SAVE RESULTS
# =====================================================

detailed_results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n======================================")
print("BIAS ANALYSIS COMPLETED")
print("======================================")

print("Detailed results saved to:")

print(OUTPUT_PATH)
