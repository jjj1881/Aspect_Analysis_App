
import time
from typing import List, Dict, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Customer Review Intelligence Dashboard",
    page_icon="🍔",
    layout="wide"
)

DETECTION_MODEL_ID = "jiangzy1881/aspect-detection-model"
SENTIMENT_MODEL_ID = "jiangzy1881/aspect-sentiment-model"

ASPECT_DISPLAY_NAMES = {
    "food": "Food",
    "menu": "Menu",
    "service": "Service",
    "staff": "Staff",
    "price": "Price",
    "place": "Place",
    "ambience": "Ambience",
    "waiting": "Waiting / Reservation",
    "miscellaneous": "Miscellaneous",
}

TEAM_MAPPING = {
    "food": "Kitchen / Food Quality Team",
    "menu": "Menu Development Team",
    "service": "Store Operations Team",
    "staff": "Staff Training Team",
    "price": "Pricing / Marketing Team",
    "place": "Store Environment Team",
    "ambience": "Store Environment Team",
    "waiting": "Operations / Reservation Team",
    "miscellaneous": "Customer Experience Team",
}


@st.cache_resource(show_spinner=True)
def load_models():
    aspect_detector = pipeline(
        task="text-classification",
        model=DETECTION_MODEL_ID,
        tokenizer=DETECTION_MODEL_ID,
        top_k=None,
        function_to_apply="sigmoid"
    )

    sentiment_classifier = pipeline(
        task="text-classification",
        model=SENTIMENT_MODEL_ID,
        tokenizer=SENTIMENT_MODEL_ID
    )

    return aspect_detector, sentiment_classifier


def flatten_pipeline_output(outputs):
    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], list):
        return outputs[0]
    return outputs


def detect_aspects(review_text: str, threshold: float = 0.50) -> Tuple[List[str], List[Dict]]:
    aspect_detector, _ = load_models()
    outputs = aspect_detector(review_text)
    outputs = flatten_pipeline_output(outputs)
    outputs = sorted(outputs, key=lambda x: x["score"], reverse=True)

    detected = [item["label"] for item in outputs if item["score"] >= threshold]

    if len(detected) == 0 and len(outputs) > 0:
        detected = [outputs[0]["label"]]

    return detected, outputs


def classify_sentiment(review_text: str, aspect: str) -> Dict:
    _, sentiment_classifier = load_models()
    aspect_display = ASPECT_DISPLAY_NAMES.get(aspect, aspect)
    input_text = f"Review: {review_text} Aspect: {aspect_display}"

    output = sentiment_classifier(input_text)
    if isinstance(output, list):
        output = output[0]

    return {
        "aspect": aspect,
        "aspect_display": aspect_display,
        "sentiment": output["label"],
        "sentiment_score": float(output["score"]),
        "sentiment_input": input_text,
    }


def assign_priority(aspect: str, sentiment: str, negative_count: int = 1) -> str:
    sentiment = str(sentiment).lower()

    if sentiment == "negative":
        if aspect in ["service", "staff", "food", "waiting"]:
            return "High"
        if negative_count >= 2:
            return "High"
        return "Medium"

    if sentiment == "neutral":
        return "Medium"

    return "Low"


def analyze_single_review(review_text: str, threshold: float = 0.50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detected_aspects, all_aspect_scores = detect_aspects(review_text, threshold=threshold)

    rows = []
    for aspect in detected_aspects:
        sentiment_result = classify_sentiment(review_text, aspect)
        rows.append(sentiment_result)

    negative_count = sum(1 for row in rows if row["sentiment"].lower() == "negative")

    for row in rows:
        row["priority"] = assign_priority(row["aspect"], row["sentiment"], negative_count)
        row["responsible_team"] = TEAM_MAPPING.get(row["aspect"], "Customer Experience Team")

    result_df = pd.DataFrame(rows)

    score_df = pd.DataFrame(all_aspect_scores)
    if not score_df.empty:
        score_df["aspect_display"] = score_df["label"].map(lambda x: ASPECT_DISPLAY_NAMES.get(x, x))
        score_df = score_df.rename(columns={"label": "aspect", "score": "aspect_score"})

    return result_df, score_df


def analyze_batch_reviews(df: pd.DataFrame, review_col: str, threshold: float = 0.50) -> pd.DataFrame:
    all_rows = []
    progress = st.progress(0)
    status = st.empty()
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        review_text = str(row[review_col])
        status.write(f"Analyzing review {i + 1} of {total}...")

        try:
            result_df, _ = analyze_single_review(review_text, threshold=threshold)

            if not result_df.empty:
                for _, pred_row in result_df.iterrows():
                    all_rows.append({
                        "review_id": i,
                        "review_text": review_text,
                        "aspect": pred_row["aspect"],
                        "aspect_display": pred_row["aspect_display"],
                        "sentiment": pred_row["sentiment"],
                        "sentiment_score": pred_row["sentiment_score"],
                        "priority": pred_row["priority"],
                        "responsible_team": pred_row["responsible_team"],
                    })

        except Exception as e:
            all_rows.append({
                "review_id": i,
                "review_text": review_text,
                "aspect": "error",
                "aspect_display": "Error",
                "sentiment": "error",
                "sentiment_score": 0.0,
                "priority": "Review manually",
                "responsible_team": "Customer Experience Team",
                "error_message": str(e),
            })

        progress.progress((i + 1) / total)

    status.empty()
    return pd.DataFrame(all_rows)


def generate_business_summary(result_df: pd.DataFrame) -> str:
    if result_df.empty:
        return "No aspect was detected. Please try a lower threshold or enter a longer review."

    negative_rows = result_df[result_df["sentiment"].str.lower() == "negative"]

    if negative_rows.empty:
        return "The review does not contain high-risk negative issues. The overall customer experience appears acceptable."

    high_priority = negative_rows[negative_rows["priority"] == "High"]

    if not high_priority.empty:
        aspects = ", ".join(high_priority["aspect_display"].tolist())
        teams = ", ".join(sorted(high_priority["responsible_team"].unique()))
        return f"The review contains high-priority negative issues related to {aspects}. It should be routed to: {teams}."

    aspects = ", ".join(negative_rows["aspect_display"].tolist())
    return f"The review contains negative feedback related to {aspects}. It should be reviewed by the responsible business team."


st.title("🍔 Customer Review Intelligence Dashboard")
st.caption("Aspect Detection + Aspect Sentiment Classification using fine-tuned Hugging Face models")

with st.sidebar:
    st.header("Model Settings")
    st.write("**Aspect Detection Model**")
    st.code(DETECTION_MODEL_ID)
    st.write("**Aspect Sentiment Model**")
    st.code(SENTIMENT_MODEL_ID)

    threshold = st.slider(
        "Aspect detection threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.markdown("---")
    st.write("Recommended threshold: **0.50**")
    st.write("Lower threshold detects more aspects; higher threshold gives stricter predictions.")

tab_intro, tab_single, tab_batch, tab_dashboard = st.tabs(
    ["Project Overview", "Single Review Analysis", "Batch CSV Analysis", "Dashboard"]
)

with tab_intro:
    st.subheader("Business Problem")
    st.write(
        "Restaurant chains receive large volumes of online customer reviews. "
        "Manually reading each review is not scalable, and overall ratings do not explain "
        "which operational areas are causing customer dissatisfaction."
    )

    st.subheader("Application Objective")
    st.write(
        "This application transforms unstructured restaurant reviews into structured, "
        "aspect-level business insights. It detects mentioned aspects, predicts sentiment "
        "for each aspect, assigns priority, and routes issues to responsible teams."
    )

    st.subheader("System Workflow")
    st.markdown(
        """
        **Review Text**  
        → **Pipeline 1: Aspect Detection**  
        → **Pipeline 2: Aspect Sentiment Classification**  
        → **Priority & Team Routing**  
        → **Management Dashboard**
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Pipeline 1: Aspect Detection**\n\n"
            "Multi-label text classification model that identifies which business aspects "
            "are mentioned in a review."
        )

    with col2:
        st.info(
            "**Pipeline 2: Aspect Sentiment**\n\n"
            "Aspect-conditioned text classification model that predicts whether the sentiment "
            "for each detected aspect is positive, neutral, or negative."
        )

with tab_single:
    st.subheader("Single Review Analysis")

    example_review = "The food was delicious, but the service was slow and the price was too high."

    review_text = st.text_area(
        "Enter a restaurant review:",
        value=example_review,
        height=120
    )

    if st.button("Analyze Review", type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Loading models and analyzing review..."):
                start = time.time()
                result_df, score_df = analyze_single_review(review_text, threshold=threshold)
                elapsed = time.time() - start

            st.success(f"Analysis completed in {elapsed:.2f} seconds.")

            st.subheader("Aspect-level Results")
            if result_df.empty:
                st.warning("No aspects detected.")
            else:
                display_cols = [
                    "aspect_display",
                    "sentiment",
                    "sentiment_score",
                    "priority",
                    "responsible_team"
                ]

                st.dataframe(result_df[display_cols], use_container_width=True)

                st.subheader("Business Summary")
                st.write(generate_business_summary(result_df))

            with st.expander("View all aspect detection scores"):
                if score_df.empty:
                    st.write("No score output available.")
                else:
                    st.dataframe(score_df, use_container_width=True)

with tab_batch:
    st.subheader("Batch CSV Analysis")
    st.write("Upload a CSV file containing a column of restaurant review texts.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        review_col = st.selectbox(
            "Select the review text column:",
            options=uploaded_df.columns.tolist()
        )

        max_rows = st.number_input(
            "Maximum number of reviews to analyze",
            min_value=1,
            max_value=len(uploaded_df),
            value=min(50, len(uploaded_df)),
            step=1
        )

        if st.button("Run Batch Analysis", type="primary"):
            batch_df = uploaded_df.head(max_rows).copy()

            with st.spinner("Running batch analysis..."):
                batch_results = analyze_batch_reviews(batch_df, review_col=review_col, threshold=threshold)

            st.session_state["batch_results"] = batch_results

            st.success("Batch analysis completed.")
            st.dataframe(batch_results, use_container_width=True)

            csv_data = batch_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Analysis Results as CSV",
                data=csv_data,
                file_name="review_analysis_results.csv",
                mime="text/csv"
            )

with tab_dashboard:
    st.subheader("Management Dashboard")

    if "batch_results" not in st.session_state:
        st.info("Please run batch analysis first in the Batch CSV Analysis tab.")
    else:
        results = st.session_state["batch_results"].copy()

        if results.empty:
            st.warning("No batch results available.")
        else:
            total_reviews = results["review_id"].nunique()
            total_aspect_records = len(results)
            negative_records = (results["sentiment"].str.lower() == "negative").sum()
            high_priority_records = (results["priority"] == "High").sum()

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Analyzed Reviews", total_reviews)
            kpi2.metric("Aspect Records", total_aspect_records)
            kpi3.metric("Negative Records", int(negative_records))
            kpi4.metric("High Priority Issues", int(high_priority_records))

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                aspect_count = (
                    results.groupby("aspect_display")
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )

                fig = px.bar(
                    aspect_count,
                    x="aspect_display",
                    y="count",
                    title="Aspect Mention Counts"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                sentiment_count = (
                    results.groupby("sentiment")
                    .size()
                    .reset_index(name="count")
                )

                fig = px.pie(
                    sentiment_count,
                    names="sentiment",
                    values="count",
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                neg_rate = (
                    results.assign(is_negative=results["sentiment"].str.lower() == "negative")
                    .groupby("aspect_display")["is_negative"]
                    .mean()
                    .reset_index(name="negative_rate")
                    .sort_values("negative_rate", ascending=False)
                )

                fig = px.bar(
                    neg_rate,
                    x="aspect_display",
                    y="negative_rate",
                    title="Negative Rate by Aspect"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                team_workload = (
                    results.groupby("responsible_team")
                    .size()
                    .reset_index(name="issue_count")
                    .sort_values("issue_count", ascending=False)
                )

                fig = px.bar(
                    team_workload,
                    x="responsible_team",
                    y="issue_count",
                    title="Responsible Team Workload"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("High Priority / Negative Issues")

            issue_table = results[
                (results["sentiment"].str.lower() == "negative") |
                (results["priority"] == "High")
            ].copy()

            if issue_table.empty:
                st.success("No high-priority negative issues found.")
            else:
                st.dataframe(
                    issue_table[
                        [
                            "review_id",
                            "review_text",
                            "aspect_display",
                            "sentiment",
                            "sentiment_score",
                            "priority",
                            "responsible_team"
                        ]
                    ],
                    use_container_width=True
                )
