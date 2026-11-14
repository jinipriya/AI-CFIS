
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def prepare_reviews(csv_path):

    df = pd.read_csv(csv_path)

    # Combine review title and content
    df["review_text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["content"].fillna("").astype(str)
    ).str.strip()

    # Remove empty reviews
    df = df[df["review_text"] != ""].copy()

    return df.reset_index(drop=True)


def perform_kmeans_clustering(df, number_of_clusters=5):

    # Convert review text into TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2
    )

    X = vectorizer.fit_transform(df["review_text"])

    # Train K-Means
    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X)

    # Add cluster assignments
    result_df = df.copy()
    result_df["Cluster"] = cluster_labels

    # Extract important keywords for each cluster
    terms = vectorizer.get_feature_names_out()

    cluster_keywords = {}

    for cluster_number in range(number_of_clusters):

        center = kmeans.cluster_centers_[cluster_number]

        top_indices = center.argsort()[-10:][::-1]

        keywords = [
            terms[index] for index in top_indices
        ]

        cluster_keywords[cluster_number] = keywords

    # Cluster counts
    cluster_summary = (
        result_df["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_summary.columns = [
        "Cluster",
        "Review Count"
    ]

    return (
        result_df,
        cluster_summary,
        cluster_keywords,
        kmeans,
        vectorizer
    )
