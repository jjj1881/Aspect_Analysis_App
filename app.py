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

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
DETECTION_MODEL_ID = "jiangzy1881/aspect-detection-model"
SENTIMENT_MODEL_ID = "jiangzy1881/aspect-sentiment-model"

# Keep the Streamlit Cloud demo responsive. Full-size experiments should be run
# in Colab/notebooks and exported to Excel for the assignment.
MAX_BATCH_ROWS = 500

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

PRIORITY_ORDER = {
    "High": 0,
    "Medium": 1,
    "Low": 2,
    "Review manually": 3,
}

SENTIMENT_ORDER = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "error": 3,
}

SENTIMENT_COLOR_MAP = {
    "negative": "#EF4444",   # red
    "neutral": "#60A5FA",    # light blue
    "positive": "#22C55E",   # green
    "error": "#9CA3AF",      # gray
}


@st.cache_resource(show_spinner=True)
def load_models():
    """Load fine-tuned Hugging Face pipelines once per Streamlit session."""
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


def normalize_label(label: str) -> str:
    """Normalize model labels for stable mapping and dashboard logic."""
    if label is None:
        return ""
    return str(label).strip().lower().replace(" ", "_")


def detect_aspects(review_text: str, threshold: float = 0.50) -> Tuple[List[str], List[Dict]]:
    """Detect all aspect labels above the threshold for multi-label classification."""
    aspect_detector, _ = load_models()
    outputs = aspect_detector(review_text)
    outputs = flatten_pipeline_output(outputs)

    cleaned_outputs = []
    for item in outputs:
        cleaned_outputs.append({
            "label": normalize_label(item.get("label", "")),
            "score": float(item.get("score", 0.0))
        })

    cleaned_outputs = sorted(cleaned_outputs, key=lambda x: x["score"], reverse=True)

    detected = [item["label"] for item in cleaned_outputs if item["score"] >= threshold]

    # If concrete aspects are detected, remove miscellaneous for cleaner business output.
    if len(detected) > 1 and "miscellaneous" in detected:
        detected = [x for x in detected if x != "miscellaneous"]

    # If nothing reaches the threshold, keep the top aspect so the app always returns a usable result.
    if len(detected) == 0 and len(cleaned_outputs) > 0:
        detected = [cleaned_outputs[0]["label"]]

    return detected, cleaned_outputs


def classify_sentiment(review_text: str, aspect: str) -> Dict:
    """Predict sentiment for a specific target aspect."""
    _, sentiment_classifier = load_models()
    aspect = normalize_label(aspect)
    aspect_display = ASPECT_DISPLAY_NAMES.get(aspect, aspect)
    input_text = f"Review: {review_text} Aspect: {aspect_display}"

    output = sentiment_classifier(input_text)
    output = flatten_pipeline_output(output)

    if isinstance(output, list) and len(output) > 0:
        output = output[0]

    if not isinstance(output, dict):
        return {
            "aspect": aspect,
            "aspect_display": aspect_display,
            "sentiment": "error",
            "sentiment_score": 0.0,
            "sentiment_input": input_text,
        }

    return {
        "aspect": aspect,
        "aspect_display": aspect_display,
        "sentiment": normalize_label(output.get("label", "error")),
        "sentiment_score": float(output.get("score", 0.0)),
        "sentiment_input": input_text,
    }


def assign_priority(aspect: str, sentiment: str, negative_count: int = 1) -> str:
    """Rule-based priority engine for restaurant management use."""
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


def sort_results_by_priority(result_df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows into an actionable management queue: High first, then Medium, Low."""
    if result_df.empty:
        return result_df

    sorted_df = result_df.copy()
    sorted_df["priority_rank"] = sorted_df["priority"].map(PRIORITY_ORDER).fillna(99).astype(int)
    sorted_df["sentiment_rank"] = (
        sorted_df["sentiment"].astype(str).str.lower().map(SENTIMENT_ORDER).fillna(99).astype(int)
    )

    sort_cols = ["priority_rank", "sentiment_rank"]
    ascending = [True, True]

    if "sentiment_score" in sorted_df.columns:
        sort_cols.append("sentiment_score")
        ascending.append(False)

    if "review_id" in sorted_df.columns:
        sort_cols.append("review_id")
        ascending.append(True)

    sorted_df = sorted_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return sorted_df


def make_download_csv(df: pd.DataFrame) -> bytes:
    """Prepare a UTF-8-SIG CSV so Excel opens it cleanly."""
    return df.to_csv(index=False).encode("utf-8-sig")


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

    result_df = sort_results_by_priority(pd.DataFrame(rows))

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
    return sort_results_by_priority(pd.DataFrame(all_rows))


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


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("🍔 Customer Review Intelligence Dashboard")
st.caption("Aspect Detection + Aspect Sentiment Classification using fine-tuned Hugging Face models")
st.info(
    "The first prediction may take longer because the fine-tuned Hugging Face models need to be loaded. "
    "Later predictions are faster due to Streamlit caching."
)

try:
    # Trigger cached loading once, with user-friendly failure handling.
    load_models()
except Exception as e:
    st.error("Failed to load Hugging Face models. Please check the model IDs and Streamlit Cloud internet access.")
    st.exception(e)
    st.stop()

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
    st.write(f"Streamlit demo batch limit: **{MAX_BATCH_ROWS} reviews**")

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
        → **Priority Queue + Management Dashboard + CSV Export**
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

    st.subheader("Business Features Added")
    st.markdown(
        """
        - Automatic **priority ranking**: High-priority issues are shown first.  
        - **Responsible team routing** based on detected aspect.  
        - **CSV export** for operational follow-up and assignment submission evidence.  
        - Dashboard filters for priority, sentiment, and aspect.
        """
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
            with st.spinner("Analyzing review..."):
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
                    "responsible_team",
                ]

                st.dataframe(result_df[display_cols], use_container_width=True)

                st.download_button(
                    label="Download Single Review Result as CSV",
                    data=make_download_csv(result_df[display_cols]),
                    file_name="single_review_analysis_result.csv",
                    mime="text/csv"
                )

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
    st.warning(
        f"For Streamlit Cloud responsiveness, this demo analyzes at most {MAX_BATCH_ROWS} reviews at a time. "
        "Run full-size experiments in Colab and report them in the project Excel file."
    )

    sample_df = pd.DataFrame({
        "review_text": [
            "The food was delicious, but the service was slow and the price was too high.",
            "The staff were friendly and the ambience was beautiful.",
            "The waiting time was too long and the table was dirty.",
            "The menu had many options, but the drinks were overpriced.",
            "The dessert was excellent and the restaurant was very comfortable.",
            "The burger was cold, and the fries were too salty.",
            "Our waiter was attentive and the food arrived quickly.",
            "The reservation process was confusing, and we waited for almost an hour.",
            "The restaurant looks modern, but the bill was surprisingly high.",
            "The pasta was tasty, although the room was noisy and crowded.",
            "The coffee was excellent, but the service at the counter was unfriendly.",
            "The steak was perfectly cooked, and the staff made us feel welcome.",
            "The place was clean, but the chairs were uncomfortable.",
            "The sushi was fresh, but the portions were too small for the price.",
            "The host was rude, and our table was not ready on time.",
            "The wine list was impressive, but the menu was hard to understand.",
            "The atmosphere was relaxing, and the music was pleasant.",
            "The soup was bland, and no one came to check on our table.",
            "The restaurant was easy to find, and the staff were very helpful.",
            "The food quality was inconsistent, but the manager handled the complaint professionally."
        ]
    })

    with st.expander("Download sample CSV for testing"):
        st.write("You can use this 20-review sample file to test batch analysis and dashboard functions.")
        st.dataframe(sample_df, use_container_width=True)
        st.download_button(
            label="Download sample_reviews.csv",
            data=make_download_csv(sample_df),
            file_name="sample_reviews.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read the uploaded CSV file: {e}")
            st.stop()

        if uploaded_df.empty:
            st.error("The uploaded CSV file is empty. Please upload a file with at least one review.")
            st.stop()

        st.write("Preview of uploaded data:")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        review_col = st.selectbox(
            "Select the review text column:",
            options=uploaded_df.columns.tolist()
        )

        max_allowed = min(len(uploaded_df), MAX_BATCH_ROWS)
        max_rows = st.number_input(
            "Maximum number of reviews to analyze",
            min_value=1,
            max_value=max_allowed,
            value=min(200, max_allowed),
            step=1,
            help=f"Limited to {MAX_BATCH_ROWS} rows for Streamlit Cloud demo performance."
        )

        if len(uploaded_df) > MAX_BATCH_ROWS:
            st.info(
                f"The uploaded file has {len(uploaded_df)} rows. Only the first {MAX_BATCH_ROWS} rows can be processed in this Streamlit demo."
            )

        if st.button("Run Batch Analysis", type="primary"):
            batch_df = uploaded_df.head(int(max_rows)).copy()
            batch_df = batch_df.dropna(subset=[review_col])
            batch_df = batch_df[batch_df[review_col].astype(str).str.strip() != ""]

            if batch_df.empty:
                st.error("No valid review text found in the selected column.")
                st.stop()

            with st.spinner("Running batch analysis..."):
                start = time.time()
                batch_results = analyze_batch_reviews(batch_df, review_col=review_col, threshold=threshold)
                elapsed = time.time() - start

            st.session_state["batch_results"] = batch_results

            st.success(f"Batch analysis completed in {elapsed:.2f} seconds. Results are sorted by priority.")

            display_cols = [
                "review_id",
                "review_text",
                "aspect_display",
                "sentiment",
                "sentiment_score",
                "priority",
                "responsible_team",
            ]
            st.dataframe(batch_results[display_cols], use_container_width=True)

            st.download_button(
                label="Download Priority-Sorted Analysis Results as CSV",
                data=make_download_csv(batch_results[display_cols]),
                file_name="priority_sorted_review_analysis_results.csv",
                mime="text/csv"
            )

with tab_dashboard:
    st.subheader("Management Dashboard")

    if "batch_results" not in st.session_state:
        st.info("Please run batch analysis first in the Batch CSV Analysis tab.")
    else:
        results = st.session_state["batch_results"].copy()
        results = sort_results_by_priority(results)

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

            st.subheader("Filter and Export Operational Queue")
            f1, f2, f3 = st.columns(3)

            with f1:
                selected_priorities = st.multiselect(
                    "Priority",
                    options=["High", "Medium", "Low", "Review manually"],
                    default=["High", "Medium", "Low", "Review manually"]
                )

            with f2:
                sentiment_options = sorted(results["sentiment"].dropna().astype(str).unique().tolist())
                selected_sentiments = st.multiselect(
                    "Sentiment",
                    options=sentiment_options,
                    default=sentiment_options
                )

            with f3:
                aspect_options = sorted(results["aspect_display"].dropna().astype(str).unique().tolist())
                selected_aspects = st.multiselect(
                    "Aspect",
                    options=aspect_options,
                    default=aspect_options
                )

            filtered_results = results[
                results["priority"].isin(selected_priorities)
                & results["sentiment"].astype(str).isin(selected_sentiments)
                & results["aspect_display"].astype(str).isin(selected_aspects)
            ].copy()
            filtered_results = sort_results_by_priority(filtered_results)

            queue_cols = [
                "review_id",
                "review_text",
                "aspect_display",
                "sentiment",
                "sentiment_score",
                "priority",
                "responsible_team",
            ]

            st.dataframe(filtered_results[queue_cols], use_container_width=True)
            st.download_button(
                label="Download Filtered Priority Queue as CSV",
                data=make_download_csv(filtered_results[queue_cols]),
                file_name="filtered_priority_queue.csv",
                mime="text/csv"
            )

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
                    title="Sentiment Distribution",
                    color="sentiment",
                    color_discrete_map=SENTIMENT_COLOR_MAP
                )
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                priority_count = (
                    results.groupby("priority")
                    .size()
                    .reset_index(name="count")
                )
                priority_count["priority_rank"] = priority_count["priority"].map(PRIORITY_ORDER).fillna(99)
                priority_count = priority_count.sort_values("priority_rank")

                fig = px.bar(
                    priority_count,
                    x="priority",
                    y="count",
                    title="Priority Distribution"
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

            col5, col6 = st.columns(2)

            with col5:
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

            with col6:
                high_priority_by_team = (
                    results[results["priority"] == "High"]
                    .groupby("responsible_team")
                    .size()
                    .reset_index(name="high_priority_count")
                    .sort_values("high_priority_count", ascending=False)
                )

                if high_priority_by_team.empty:
                    st.info("No high-priority issues found in the current batch.")
                else:
                    fig = px.bar(
                        high_priority_by_team,
                        x="responsible_team",
                        y="high_priority_count",
                        title="High Priority Issues by Team"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("High Priority / Negative Issues")

            issue_table = results[
                (results["sentiment"].str.lower() == "negative") |
                (results["priority"] == "High")
            ].copy()
            issue_table = sort_results_by_priority(issue_table)

            if issue_table.empty:
                st.success("No high-priority negative issues found.")
            else:
                st.dataframe(issue_table[queue_cols], use_container_width=True)
                st.download_button(
                    label="Download High-Priority / Negative Issues as CSV",
                    data=make_download_csv(issue_table[queue_cols]),
                    file_name="high_priority_negative_issues.csv",
                    mime="text/csv"
                )
