
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_fairness_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "Accuracy": round(
            accuracy_score(y_true, y_pred) * 100, 2
        ),
        "Precision": round(
            precision_score(y_true, y_pred, zero_division=0) * 100, 2
        ),
        "Recall": round(
            recall_score(y_true, y_pred, zero_division=0) * 100, 2
        ),
        "F1 Score": round(
            f1_score(y_true, y_pred, zero_division=0) * 100, 2
        ),
        "False Positive Rate": round(fpr * 100, 2),
        "False Negative Rate": round(fnr * 100, 2)
    }


def analyze_review_length(df, y_true, y_pred):

    result = df.copy()

    result["Actual"] = list(y_true)
    result["Predicted"] = list(y_pred)

    result["Word Count"] = (
        result["review_text"].fillna("").astype(str).str.split().str.len()
    )

    result["Review Length"] = pd.cut(
        result["Word Count"],
        bins=[-1, 20, 50, float("inf")],
        labels=["Short", "Medium", "Long"]
    )

    rows = []

    for group in ["Short", "Medium", "Long"]:

        group_df = result[
            result["Review Length"] == group
        ]

        if group_df.empty:
            continue

        metrics = calculate_fairness_metrics(
            group_df["Actual"],
            group_df["Predicted"]
        )

        metrics["Review Length"] = group
        metrics["Review Count"] = len(group_df)

        rows.append(metrics)

    return pd.DataFrame(rows), result
