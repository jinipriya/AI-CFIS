
import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import joblib
import re
import sys
from pathlib import Path
from collections import Counter

# Add project root to Python import path
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from modules.ml_comparison import (
    prepare_dataset,
    train_and_compare_models
)
from modules.kmeans_clustering import (
    prepare_reviews,
    perform_kmeans_clustering
)
from modules.ocr_analysis import extract_text_from_image
from modules.bias_analysis import (
    calculate_fairness_metrics,
    analyze_review_length
)
# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

BERT_MODEL_PATH = BASE_DIR / "models" / "distilbert_sentiment"

LR_MODEL_PATH = BASE_DIR / "models" / "logistic_regression" / "sentiment_model.pkl"

TFIDF_PATH = BASE_DIR / "models" / "logistic_regression" / "tfidf_vectorizer.pkl"


# =========================================================
# 2. CUSTOM LIGHT THEME
# =========================================================


st.markdown("""
<style>

/* =========================================
   1. MAIN APPLICATION
========================================= */

.stApp {
    background-color: #F4F8FF !important;
    color: #1E293B !important;
}

.block-container {
    padding: 2rem 2.5rem;
    max-width: 1500px;
}

/* Main headings */
h1 {
    color: #172554 !important;
    font-weight: 750 !important;
    letter-spacing: -0.5px;
}

h2, h3 {
    color: #1E3A5F !important;
    font-weight: 650 !important;
}

/* General text */
p,
label,
.stMarkdown,
[data-testid="stWidgetLabel"] {
    color: #334155 !important;
}


/* =========================================
   2. SIDEBAR
========================================= */

[data-testid="stSidebar"] {
    background-color: #E8F2FF !important;
    border-right: 1px solid #D6E6FA;
}

[data-testid="stSidebar"] * {
    color: #1E3A5F !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #172554 !important;
}


/* =========================================
   3. TEXT AREA
========================================= */

[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #94A3B8 !important;
}

[data-testid="stTextArea"] textarea:focus {
    border: 1px solid #60A5FA !important;
    box-shadow: 0 0 0 2px #DBEAFE !important;
}


/* =========================================
   4. SELECT BOX - LIGHT BLUE
========================================= */

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #EAF3FF !important;
    color: #1E3A8A !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 10px !important;
    min-height: 48px;
}

[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #1E3A8A !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #2563EB !important;
}

/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="menu"] * {
    color: #1E3A8A !important;
}

[data-baseweb="menu"] li:hover {
    background-color: #DBEAFE !important;
}


/* =========================================
   5. BUTTONS
========================================= */

.stButton > button,
.stDownloadButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.65rem 1.4rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}

.stButton > button p,
.stDownloadButton > button p {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #1D4ED8 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.18);
}


/* =========================================
   6. SUCCESS MESSAGE
========================================= */

[data-testid="stAlert"][kind="success"] {
    background-color: #DCFCE7 !important;
    border: 1px solid #86EFAC !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"][kind="success"] * {
    color: #166534 !important;
}


/* =========================================
   7. INFORMATION MESSAGE
========================================= */

[data-testid="stAlert"][kind="info"] {
    background-color: #DBEAFE !important;
    border: 1px solid #93C5FD !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"][kind="info"] * {
    color: #1E40AF !important;
}


/* =========================================
   8. ERROR MESSAGE
========================================= */

[data-testid="stAlert"][kind="error"] {
    background-color: #FEE2E2 !important;
    border: 1px solid #FCA5A5 !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"][kind="error"] * {
    color: #991B1B !important;
}


/* =========================================
   9. WARNING MESSAGE
========================================= */

[data-testid="stAlert"][kind="warning"] {
    background-color: #FEF3C7 !important;
    border: 1px solid #FCD34D !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"][kind="warning"] * {
    color: #92400E !important;
}


/* =========================================
   10. KPI / METRIC CARDS
========================================= */

[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE9F8 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 3px 12px rgba(30, 64, 175, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #172554 !important;
    font-weight: 750 !important;
}


/* =========================================
   11. FILE UPLOADER
========================================= */

[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px dashed #93C5FD !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

[data-testid="stFileUploader"] * {
    color: #334155 !important;
}


/* =========================================
   12. DATA TABLE
========================================= */

[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE9F8 !important;
    border-radius: 10px !important;
}


/* =========================================
   13. PROGRESS BAR
========================================= */

[data-testid="stProgressBar"] > div > div > div {
    background-color: #2563EB !important;
}


/* =========================================
   14. DIVIDERS
========================================= */

hr {
    border-color: #DCE6F2 !important;
}


/* =========================================
   15. FOOTER
========================================= */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)



# =========================================================
# 3. MODEL LOADING
# =========================================================

@st.cache_resource
def load_bert_model():

    if not BERT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"DistilBERT model folder not found: {BERT_MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(BERT_MODEL_PATH)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(BERT_MODEL_PATH)
    )

    model.to("cpu")
    model.eval()

    return tokenizer, model


@st.cache_resource
def load_logistic_model():

    if not LR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Logistic Regression model not found: {LR_MODEL_PATH}"
        )

    if not TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {TFIDF_PATH}"
        )

    lr_model = joblib.load(LR_MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)

    return lr_model, tfidf_vectorizer


# =========================================================
# 4. SENTIMENT PREDICTION FUNCTIONS
# =========================================================

def predict_distilbert(text):

    tokenizer, model = load_bert_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][predicted_class].item()

    label_map = {
        0: "Negative",
        1: "Positive"
    }

    sentiment = label_map[predicted_class]

    return sentiment, confidence


def predict_logistic(text):

    lr_model, tfidf_vectorizer = load_logistic_model()

    transformed_text = tfidf_vectorizer.transform([text])

    prediction = int(lr_model.predict(transformed_text)[0])

    probabilities = lr_model.predict_proba(transformed_text)[0]

    confidence = float(probabilities[prediction])

    sentiment = "Positive" if prediction == 1 else "Negative"

    return sentiment, confidence


def predict_sentiment(text, selected_model):

    if selected_model == "DistilBERT":
        return predict_distilbert(text)

    return predict_logistic(text)


# =========================================================
# 5. SESSION STATE
# =========================================================

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = pd.DataFrame()


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("## Feedback Intelligence")

    st.caption("AI-Powered Customer Feedback Analysis")

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Select Module",
        [
            "Dashboard",
            "Sentiment Prediction",
            "Bulk Feedback Analysis",
            "Feedback Intelligence",
            "ML Model Comparison",
            "K-Means Clustering",
            "OCR Sentiment Analysis",
            "Bias and Fairness Analysis"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("AI-Based Customer Feedback Intelligence System")
    st.caption("Powered by Logistic Regression and DistilBERT")


# =========================================================
# 7. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("Customer Feedback Intelligence")

    st.caption(
        "Understand customer opinions and discover insights using Artificial Intelligence."
    )

    results = st.session_state.analysis_results

    st.markdown("### Performance Overview")

    if not results.empty:

        total_reviews = len(results)

        positive_reviews = (
            results["Sentiment"] == "Positive"
        ).sum()

        negative_reviews = (
            results["Sentiment"] == "Negative"
        ).sum()

        positive_percentage = (
            positive_reviews / total_reviews * 100
        )

        negative_percentage = (
            negative_reviews / total_reviews * 100
        )

    else:

        total_reviews = 10000
        positive_reviews = 5000
        negative_reviews = 5000

        positive_percentage = 50
        negative_percentage = 50

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        f"{total_reviews:,}"
    )

    col2.metric(
        "Positive Reviews",
        f"{positive_reviews:,}",
        f"{positive_percentage:.1f}%"
    )

    col3.metric(
        "Negative Reviews",
        f"{negative_reviews:,}",
        f"{negative_percentage:.1f}%"
    )

    col4.metric(
        "DistilBERT Accuracy",
        "93.25%"
    )

    if results.empty:
        st.caption(
            "Showing the original 10,000-review dataset snapshot. "
            "Upload and analyze feedback to view your actual analysis results here."
        )
    else:
        st.caption("Dashboard metrics are based on your latest bulk analysis.")

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("### Sentiment Distribution")

        sentiment_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [positive_reviews, negative_reviews]
        })

        fig = px.pie(
            sentiment_data,
            names="Sentiment",
            values="Count",
            hole=0.58,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#16A34A",
                "Negative": "#EF4444"
            }
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#24324B",
            margin=dict(t=20, b=20, l=20, r=20),
            legend_title_text=""
        )

        st.plotly_chart(fig, width="stretch")

    with right:

        st.markdown("### Model Performance")

        model_data = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "DistilBERT"
            ],
            "Accuracy": [
                88.85,
                93.25
            ]
        })

        fig = px.bar(
            model_data,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            color="Model",
            color_discrete_sequence=[
                "#93C5FD",
                "#2563EB"
            ]
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#24324B",
            showlegend=False,
            yaxis=dict(
                range=[0, 105],
                title="Accuracy (%)"
            ),
            margin=dict(t=30, b=20, l=20, r=20)
        )

        st.plotly_chart(fig, width="stretch")


# =========================================================
# 8. SENTIMENT PREDICTION
# =========================================================

elif page == "Sentiment Prediction":

    st.title("Sentiment Prediction")

    st.caption(
        "Enter customer feedback and identify whether the sentiment is positive or negative."
    )

    st.markdown("### Analyze Customer Feedback")

    review = st.text_area(
        "Customer Feedback",
        placeholder="Example: The product quality is excellent. I am very happy with the service.",
        height=160
    )

    model_choice = st.selectbox(
        "Select AI Model",
        [
            "DistilBERT",
            "Logistic Regression"
        ]
    )

    if st.button("Analyze Sentiment", type="primary"):

        if not review.strip():

            st.warning("Please enter customer feedback.")

        else:

            try:

                with st.spinner(
                    f"Analyzing feedback using {model_choice}..."
                ):

                    sentiment, confidence = predict_sentiment(
                        review,
                        model_choice
                    )

                st.divider()

                st.markdown("### Prediction Result")

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    if sentiment == "Positive":

                        st.success("Positive Sentiment")

                    else:

                        st.error("Negative Sentiment")

                with result_col2:

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                st.markdown("### Customer Feedback")

                st.info(review)

                st.caption(
                    f"Prediction generated using {model_choice}"
                )

            except Exception as error:

                st.error(
                    f"Unable to generate prediction: {error}"
                )


# =========================================================
# 9. BULK FEEDBACK ANALYSIS
# =========================================================

elif page == "Bulk Feedback Analysis":

    st.title("Bulk Feedback Analysis")

    st.caption(
        "Upload customer reviews in CSV or Excel format and analyze them in bulk."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer Feedback File",
        type=["csv", "xlsx"]
    )

    if uploaded_file:

        try:

            if uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(uploaded_file)

            st.success("File uploaded successfully!")

            st.markdown("### Dataset Preview")

            st.dataframe(
                df.head(10),
                width="stretch"
            )

            if len(df.columns) == 0:
                st.error("The uploaded file has no columns.")
                st.stop()

            review_column = st.selectbox(
                "Select the customer feedback column",
                df.columns.tolist()
            )

            bulk_model = st.selectbox(
                "Select AI Model",
                [
                    "DistilBERT",
                    "Logistic Regression"
                ],
                key="bulk_model"
            )

            if st.button(
                "Analyze All Reviews",
                type="primary"
            ):

                reviews = (
                    df[review_column]
                    .fillna("")
                    .astype(str)
                    .tolist()
                )

                valid_indices = [
                    i for i, review in enumerate(reviews)
                    if review.strip()
                ]

                if not valid_indices:

                    st.warning(
                        "No valid customer reviews found."
                    )

                else:

                    sentiments = [None] * len(reviews)
                    confidences = [None] * len(reviews)

                    progress = st.progress(0)

                    status = st.empty()

                    try:

                        if bulk_model == "DistilBERT":

                            tokenizer, model = load_bert_model()

                            batch_size = 16

                            for start in range(
                                0,
                                len(valid_indices),
                                batch_size
                            ):

                                batch_indices = valid_indices[
                                    start:start + batch_size
                                ]

                                batch_reviews = [
                                    reviews[i] for i in batch_indices
                                ]

                                inputs = tokenizer(
                                    batch_reviews,
                                    return_tensors="pt",
                                    truncation=True,
                                    padding=True,
                                    max_length=128
                                )

                                with torch.inference_mode():

                                    outputs = model(**inputs)

                                    probs = torch.softmax(
                                        outputs.logits,
                                        dim=1
                                    )

                                predictions = torch.argmax(
                                    probs,
                                    dim=1
                                )

                                for j, original_index in enumerate(
                                    batch_indices
                                ):

                                    prediction = int(
                                        predictions[j].item()
                                    )

                                    sentiments[original_index] = (
                                        "Positive"
                                        if prediction == 1
                                        else "Negative"
                                    )

                                    confidences[original_index] = float(
                                        probs[j][prediction].item()
                                    )

                                progress.progress(
                                    min(
                                        1.0,
                                        (start + len(batch_indices))
                                        / len(valid_indices)
                                    )
                                )

                                status.write(
                                    f"Analyzed {min(start + batch_size, len(valid_indices))} "
                                    f"of {len(valid_indices)} reviews..."
                                )

                        else:

                            lr_model, vectorizer = load_logistic_model()

                            valid_reviews = [
                                reviews[i] for i in valid_indices
                            ]

                            transformed = vectorizer.transform(
                                valid_reviews
                            )

                            predictions = lr_model.predict(transformed)

                            probabilities = lr_model.predict_proba(
                                transformed
                            )

                            for j, original_index in enumerate(
                                valid_indices
                            ):

                                prediction = int(predictions[j])

                                sentiments[original_index] = (
                                    "Positive"
                                    if prediction == 1
                                    else "Negative"
                                )

                                confidences[original_index] = float(
                                    probabilities[j][prediction]
                                )

                            progress.progress(1.0)

                        status.success("Analysis completed!")

                        result_df = df.copy()

                        result_df["Sentiment"] = sentiments

                        result_df["Confidence"] = [
                            round(c * 100, 2)
                            if c is not None else None
                            for c in confidences
                        ]

                        st.session_state.analysis_results = result_df

                        st.markdown("### Analysis Results")

                        total = len(valid_indices)

                        positive = sentiments.count("Positive")
                        negative = sentiments.count("Negative")

                        c1, c2, c3 = st.columns(3)

                        c1.metric("Analyzed Reviews", f"{total:,}")
                        c2.metric("Positive", f"{positive:,}")
                        c3.metric("Negative", f"{negative:,}")

                        st.dataframe(
                            result_df,
                            width="stretch"
                        )

                        csv_data = result_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download Analysis Results",
                            data=csv_data,
                            file_name="customer_feedback_analysis.csv",
                            mime="text/csv"
                        )

                    except Exception as error:

                        st.error(
                            f"Bulk analysis failed: {error}"
                        )

        except Exception as error:

            st.error(
                f"Unable to read uploaded file: {error}"
            )


# =========================================================
# 10. FEEDBACK INTELLIGENCE
# =========================================================

elif page == "Feedback Intelligence":

    st.title("Feedback Intelligence")

    st.caption(
        "Discover recurring customer concerns and positive experiences."
    )

    results = st.session_state.analysis_results

    if results.empty:

        st.info(
            "Please analyze a customer feedback file in the Bulk Feedback Analysis module first."
        )

    else:

        st.markdown("### Customer Sentiment Insights")

        positive_reviews = results[
            results["Sentiment"] == "Positive"
        ]

        negative_reviews = results[
            results["Sentiment"] == "Negative"
        ]

        c1, c2 = st.columns(2)

        c1.metric(
            "Positive Experiences",
            len(positive_reviews)
        )

        c2.metric(
            "Customer Complaints",
            len(negative_reviews)
        )

        st.divider()

        st.markdown("### Common Feedback Keywords")

        stop_words = {
            "the", "and", "for", "was", "are", "is",
            "this", "that", "with", "have", "has",
            "had", "not", "but", "you", "your",
            "from", "they", "their", "very", "its",
            "product", "would", "could", "about",
            "been", "were", "will", "all", "one",
            "what", "when", "which", "there", "than",
            "them", "then", "also", "just", "really",
            "into", "out", "our", "can", "get",
            "got", "my", "me", "i", "it", "a", "an",
            "to", "of", "in", "on", "at", "as",
            "be", "by", "or", "if", "so", "no"
        }

        def extract_keywords(series):

            words = []

            for review in series.fillna("").astype(str):

                tokens = re.findall(
                    r"\b[a-zA-Z]{3,}\b",
                    review.lower()
                )

                words.extend([
                    word for word in tokens
                    if word not in stop_words
                ])

            return Counter(words).most_common(10)

        left, right = st.columns(2)

        with left:

            st.markdown("#### Positive Feedback Keywords")

            positive_keywords = extract_keywords(
                positive_reviews.iloc[:, 0]
                if False else positive_reviews[
                    positive_reviews.columns[0]
                ]
            ) if not positive_reviews.empty else []

            if positive_keywords:

                positive_df = pd.DataFrame(
                    positive_keywords,
                    columns=["Keyword", "Frequency"]
                )

                fig = px.bar(
                    positive_df,
                    x="Frequency",
                    y="Keyword",
                    orientation="h",
                    color_discrete_sequence=["#16A34A"]
                )

                fig.update_layout(
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font_color="#24324B",
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(fig, width="stretch")

            else:

                st.info("No positive feedback keywords available.")

        with right:

            st.markdown("#### Negative Feedback Keywords")

            negative_keywords = extract_keywords(
                negative_reviews[
                    negative_reviews.columns[0]
                ]
            ) if not negative_reviews.empty else []

            if negative_keywords:

                negative_df = pd.DataFrame(
                    negative_keywords,
                    columns=["Keyword", "Frequency"]
                )

                fig = px.bar(
                    negative_df,
                    x="Frequency",
                    y="Keyword",
                    orientation="h",
                    color_discrete_sequence=["#EF4444"]
                )

                fig.update_layout(
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font_color="#24324B",
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(fig, width="stretch")

            else:

                st.info("No negative feedback keywords available.")

        st.divider()

        st.markdown("### Negative Customer Feedback")

        if not negative_reviews.empty:

            st.dataframe(
                negative_reviews,
                width="stretch"
            )

        else:

            st.success(
                "No negative reviews found in the analyzed dataset."
            )


# =========================================================
# ML MODEL COMPARISON
# =========================================================

elif page == "ML Model Comparison":

    st.title("ML Model Comparison")

    st.caption(
        "Compare traditional machine learning algorithms "
        "for customer sentiment classification."
    )

    st.info(
        "All algorithms use the same dataset, "
        "stratified train-test split and TF-IDF features."
    )

    if st.button("Run Model Comparison", type="primary"):

        BASE_DIR = Path(__file__).resolve().parent.parent

        DATASET_PATH = (
            BASE_DIR
            / "data"
            / "raw"
            / "customer_feedback.csv"
        )

        try:

            with st.spinner("Preparing dataset..."):

                (
                    X_train_tfidf,
                    X_test_tfidf,
                    y_train,
                    y_test,
                    tfidf
                ) = prepare_dataset(DATASET_PATH)

            st.success("Dataset preparation completed!")

            with st.spinner(
                "Training and evaluating ML models..."
            ):

                results, matrices, trained_models = (
                    train_and_compare_models(
                        X_train_tfidf,
                        X_test_tfidf,
                        y_train,
                        y_test
                    )
                )

            st.session_state.ml_results = results
            st.session_state.ml_matrices = matrices

            st.success("Model comparison completed!")

        except Exception as error:

            st.error(f"Model comparison failed: {error}")

    # Display results after execution
    if "ml_results" in st.session_state:

        results = st.session_state.ml_results
        matrices = st.session_state.ml_matrices

        st.divider()

        st.markdown("### Model Performance Metrics")

        st.dataframe(
            results,
            hide_index=True,
            width="stretch"
        )

        st.markdown("### Accuracy Comparison")

        fig = px.bar(
            results,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            color="Model",
            title="Traditional ML Model Accuracy",
            color_discrete_sequence=[
                "#93C5FD",
                "#2563EB",
                "#60A5FA",
                "#1D4ED8"
            ]
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#24324B",
            yaxis=dict(
                range=[0, 105],
                title="Accuracy (%)"
            ),
            showlegend=False
        )

        st.plotly_chart(fig, width="stretch")

        st.markdown("### Confusion Matrix")

        selected_model = st.selectbox(
            "Select Model",
            results["Model"].tolist()
        )

        matrix = matrices[selected_model]

        fig = px.imshow(
            matrix,
            text_auto=True,
            labels=dict(
                x="Predicted Label",
                y="Actual Label",
                color="Count"
            ),
            x=["Negative", "Positive"],
            y=["Negative", "Positive"],
            color_continuous_scale="Blues",
            title=f"{selected_model} Confusion Matrix"
        )

        st.plotly_chart(fig, width="stretch")

        st.markdown("### Best Performing Model")

        best_model = results.iloc[0]

        st.success(
            f"Best model: {best_model['Model']} "
            f"with {best_model['Accuracy']:.2f}% accuracy."
        )

elif page == "K-Means Clustering":

    st.title("Customer Feedback Clustering")
    st.write(
        "Discover common themes and categories in customer feedback "
        "using the K-Means unsupervised learning algorithm."
    )

    st.info(
        "K-Means groups similar reviews based on text patterns. "
        "Cluster numbers are automatically assigned and do not represent rankings."
    )

    number_of_clusters = st.slider(
        "Select Number of Clusters",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )

    if st.button("Run Customer Feedback Clustering", type="primary"):

        with st.spinner("Analyzing customer reviews and identifying clusters..."):

            dataset_path = BASE_DIR / "data" / "raw" / "customer_feedback.csv"

            df = prepare_reviews(dataset_path)

            (
                result_df,
                cluster_summary,
                cluster_keywords,
                kmeans,
                vectorizer
            ) = perform_kmeans_clustering(
                df,
                number_of_clusters=number_of_clusters
            )

            st.session_state["kmeans_results"] = result_df
            st.session_state["kmeans_summary"] = cluster_summary
            st.session_state["kmeans_keywords"] = cluster_keywords

        st.success("Customer feedback clustering completed successfully!")

    if "kmeans_results" in st.session_state:

        result_df = st.session_state["kmeans_results"]
        cluster_summary = st.session_state["kmeans_summary"]
        cluster_keywords = st.session_state["kmeans_keywords"]

        st.subheader("Cluster Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Reviews Analyzed", f"{len(result_df):,}")

        with col2:
            st.metric("Number of Clusters", len(cluster_summary))

        st.subheader("Review Distribution")

        fig = px.bar(
            cluster_summary,
            x="Cluster",
            y="Review Count",
            color="Cluster",
            text="Review Count",
            title="Customer Reviews by Cluster"
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Cluster",
            yaxis_title="Number of Reviews"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Keywords")

        for cluster, keywords in cluster_keywords.items():

            with st.expander(f"Cluster {cluster} - {cluster_summary.loc[cluster, 'Review Count']:,} Reviews"):

                st.write("**Top Keywords:**")
                st.write(", ".join(keywords))

                st.write("**Sample Customer Reviews:**")

                samples = result_df[
                    result_df["Cluster"] == cluster
                ]["review_text"].head(5)

                for review in samples:
                    st.markdown(f"- {review[:300]}")

        st.subheader("Download Cluster Results")

        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Cluster Results (CSV)",
            data=csv_data,
            file_name="customer_feedback_clusters.csv",
            mime="text/csv"
        )

elif page == "OCR Sentiment Analysis":

    st.title("OCR-Based Customer Feedback Analysis")

    st.caption(
        "Upload an image containing customer feedback, extract the text "
        "using OCR, and predict its sentiment using AI."
    )

    st.markdown("### Upload Customer Review Image")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        key="ocr_image"
    )

    ocr_model_choice = st.selectbox(
        "Select AI Model",
        ["DistilBERT", "Logistic Regression"],
        key="ocr_model_choice"
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded Customer Review",
            use_container_width=True
        )

        if st.button("Extract Text and Analyze Sentiment", type="primary"):

            try:

                with st.spinner("Extracting text from image..."):

                    extracted_text = extract_text_from_image(
                        uploaded_image
                    )

                if not extracted_text.strip():

                    st.warning(
                        "No readable text was detected. "
                        "Please upload a clearer image."
                    )

                else:

                    st.markdown("### Extracted Customer Feedback")

                    extracted_text = st.text_area(
                        "Review Text",
                        value=extracted_text,
                        height=180,
                        key="ocr_extracted_text"
                    )

                    with st.spinner(
                        f"Analyzing sentiment using {ocr_model_choice}..."
                    ):

                        sentiment, confidence = predict_sentiment(
                            extracted_text,
                            ocr_model_choice
                        )

                    st.divider()

                    st.markdown("### Sentiment Prediction")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:

                        if sentiment == "Positive":
                            st.success("Positive Sentiment")
                        else:
                            st.error("Negative Sentiment")

                    with result_col2:

                        st.metric(
                            "Prediction Confidence",
                            f"{confidence * 100:.2f}%"
                        )

                    st.caption(
                        f"Prediction generated using {ocr_model_choice}"
                    )

            except Exception as error:

                st.error(
                    f"OCR sentiment analysis failed: {error}"
                )

elif page == "Bias and Fairness Analysis":

    st.title("Bias and Fairness Analysis")
    st.caption(
        "Evaluate sentiment prediction performance across different "
        "customer review lengths."
    )

    bias_path = BASE_DIR / "bias_analysis_results.csv"

    if not bias_path.exists():
        st.warning(
            "Bias analysis results not found. "
            "Please run test_analysis_bias.py first."
        )
        st.stop()

    bias_df = pd.read_csv(bias_path)

    # Overall performance metrics
    overall_metrics = calculate_fairness_metrics(
        bias_df["Actual"],
        bias_df["Predicted"]
    )

    st.subheader("Overall Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f'{overall_metrics["Accuracy"]}%')
    col2.metric("Precision", f'{overall_metrics["Precision"]}%')
    col3.metric("Recall", f'{overall_metrics["Recall"]}%')
    col4.metric("F1 Score", f'{overall_metrics["F1 Score"]}%')

    col5, col6 = st.columns(2)

    col5.metric(
        "False Positive Rate",
        f'{overall_metrics["False Positive Rate"]}%'
    )
    col6.metric(
        "False Negative Rate",
        f'{overall_metrics["False Negative Rate"]}%'
    )

    # Review length analysis
    st.divider()
    st.subheader("Performance by Review Length")

    length_metrics, _ = analyze_review_length(
        bias_df[["review_text"]],
        bias_df["Actual"],
        bias_df["Predicted"]
    )

    st.dataframe(
        length_metrics,
        use_container_width=True,
        hide_index=True
    )

    # Accuracy comparison
    st.subheader("Accuracy Across Review Groups")

    accuracy_chart = px.bar(
        length_metrics,
        x="Review Length",
        y="Accuracy",
        text="Accuracy",
        title="Model Accuracy by Review Length",
        labels={
            "Review Length": "Review Category",
            "Accuracy": "Accuracy (%)"
        }
    )

    accuracy_chart.update_traces(texttemplate="%{text}%", textposition="outside")
    accuracy_chart.update_layout(yaxis_range=[0, 110])

    st.plotly_chart(accuracy_chart, use_container_width=True)

    # Error rate comparison
    st.subheader("False Prediction Rates")

    error_chart = px.bar(
        length_metrics,
        x="Review Length",
        y=["False Positive Rate", "False Negative Rate"],
        barmode="group",
        title="False Positive vs False Negative Rates",
        labels={
            "value": "Rate (%)",
            "Review Length": "Review Category",
            "variable": "Error Type"
        }
    )

    st.plotly_chart(error_chart, use_container_width=True)

    # Correct vs incorrect predictions
    st.subheader("Prediction Error Overview")

    correct_count = int(
        (bias_df["Actual"] == bias_df["Predicted"]).sum()
    )
    incorrect_count = len(bias_df) - correct_count

    pie_data = pd.DataFrame({
        "Prediction Result": ["Correct", "Incorrect"],
        "Count": [correct_count, incorrect_count]
    })

    pie_chart = px.pie(
        pie_data,
        names="Prediction Result",
        values="Count",
        title="Correct vs Incorrect Predictions",
        hole=0.4
    )

    st.plotly_chart(pie_chart, use_container_width=True)

    st.write(f"**Total reviews analysed:** {len(bias_df):,}")
    st.write(f"**Correct predictions:** {correct_count:,}")
    st.write(f"**Incorrect predictions:** {incorrect_count:,}")

    # Detailed results
    st.divider()
    st.subheader("Detailed Prediction Results")

    display_columns = [
        "review_text",
        "Actual",
        "Predicted",
        "Review Length",
        "Word Count"
    ]

    if "Prediction Result" in bias_df.columns:
        display_columns.append("Prediction Result")

    st.dataframe(
        bias_df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # Download results
    st.download_button(
        label="Download Bias Analysis Results",
        data=bias_df.to_csv(index=False).encode("utf-8"),
        file_name="bias_analysis_results.csv",
        mime="text/csv"
    )

    st.info(
        "Note: This analysis evaluates model performance across review "
        "length categories. It does not establish demographic fairness "
        "because the dataset does not contain demographic attributes."
    )

# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Customer Feedback Intelligence System | "
    "Logistic Regression + DistilBERT"
)
