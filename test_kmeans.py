
from pathlib import Path

from modules.kmeans_clustering import (
    prepare_reviews,
    perform_kmeans_clustering
)


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "customer_feedback.csv"
)


# Load reviews
df = prepare_reviews(DATASET_PATH)

print("Dataset loaded successfully.")
print("Total reviews:", len(df))


# Perform clustering
(
    result_df,
    cluster_summary,
    cluster_keywords,
    kmeans,
    vectorizer
) = perform_kmeans_clustering(
    df,
    number_of_clusters=5
)


print("\nCLUSTER SUMMARY")
print(cluster_summary.to_string(index=False))


print("\nCLUSTER KEYWORDS")

for cluster, keywords in cluster_keywords.items():

    print(f"\nCluster {cluster}:")
    print(", ".join(keywords))


print("\nSAMPLE REVIEWS")

for cluster in sorted(result_df["Cluster"].unique()):

    print(f"\nCluster {cluster}:")

    samples = result_df[
        result_df["Cluster"] == cluster
    ]["review_text"].head(3)

    for review in samples:
        print("-", review[:250])


# Save experimental results
OUTPUT_PATH = BASE_DIR / "kmeans_results.csv"

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:", OUTPUT_PATH)
