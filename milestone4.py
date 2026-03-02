import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud
import numpy as np

st.set_page_config(
    page_title="ReviewSense Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
font-size: 3rem;
color: #1f77b4;
text-align: center;
margin-bottom: 2rem;
}
.metric-card {
background-color: #f0f2f6;
padding: 1.5rem;
border-radius: 12px;
text-align: center;
box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Milestone2_Sentiment_Results_new.csv")
    df["sentiment"] = df["sentiment"].str.capitalize()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data
def load_keywords():
    try:
        return pd.read_csv("Milestone3_Keyword_Insights.csv")
    except:
        return pd.DataFrame()

df = load_data()
keywords_df = load_keywords()

st.sidebar.header("🔍 Filters")

sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    options=["Positive", "Negative", "Neutral"],
    default=["Positive", "Negative", "Neutral"]
)

product_filter = st.sidebar.multiselect(
    "Select Product",
    options=sorted(df["product"].unique()),
    default=sorted(df["product"].unique())
)

st.sidebar.subheader("📅 Date Range")

default_start = df["date"].min().date() if pd.notna(df["date"].min()) else datetime(2025, 1, 1).date()
default_end = df["date"].max().date() if pd.notna(df["date"].max()) else datetime(2025, 12, 31).date()

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=default_start)
end_date = col2.date_input("End Date", value=default_end)

filtered_df = df[
    (df["sentiment"].isin(sentiment_filter)) &
    (df["product"].isin(product_filter)) &
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
].copy()

st.markdown(
    '<h1 class="main-header">📊 ReviewSense – Customer Feedback Dashboard</h1>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

total_reviews = len(filtered_df)
pos_count = len(filtered_df[filtered_df["sentiment"] == "Positive"])
neg_count = len(filtered_df[filtered_df["sentiment"] == "Negative"])
neu_count = len(filtered_df[filtered_df["sentiment"] == "Neutral"])

pos_pct = (pos_count / total_reviews * 100) if total_reviews > 0 else 0
neg_pct = (neg_count / total_reviews * 100) if total_reviews > 0 else 0
neu_pct = (neu_count / total_reviews * 100) if total_reviews > 0 else 0

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Reviews", total_reviews)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Positive", f"{pos_pct:.1f}%", delta=f"{pos_count} reviews")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Negative", f"{neg_pct:.1f}%", delta=f"{neg_count} reviews")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Neutral", f"{neu_pct:.1f}%", delta=f"{neu_count} reviews")
    st.markdown('</div>', unsafe_allow_html=True)

st.subheader("😊 Sentiment Distribution")

if not filtered_df.empty:
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    counts = filtered_df["sentiment"].value_counts()
    colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}
    bars = ax1.bar(counts.index, counts.values, color=[colors.get(s, "gray") for s in counts.index])
    ax1.set_xlabel("Sentiment")
    ax1.set_ylabel("Number of Reviews")
    ax1.set_title("Overall Sentiment Breakdown")
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha="center")
    st.pyplot(fig1)

st.subheader("📱 Product-wise Sentiment")

if not filtered_df.empty:
    product_sent = filtered_df.groupby("product")["sentiment"].value_counts().unstack(fill_value=0)
    product_sent["Total"] = product_sent.sum(axis=1)
    product_sent["Positive %"] = (product_sent.get("Positive", 0) / product_sent["Total"] * 100).round(1)
    product_sent = product_sent.sort_values("Positive %", ascending=False)
    st.dataframe(product_sent, use_container_width=True)

    fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
    sns.heatmap(product_sent[["Positive", "Negative", "Neutral"]], annot=True, fmt="d", cmap="RdYlGn", ax=ax_hm)
    ax_hm.set_title("Product Sentiment Heatmap")
    st.pyplot(fig_hm)

st.subheader("📈 Sentiment Trends Over Time")

if not filtered_df.empty:
    filtered_df["month"] = filtered_df["date"].dt.to_period("M")
    trend = filtered_df.groupby(["month", "sentiment"]).size().unstack(fill_value=0)

    fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
    for col in trend.columns:
        ax_trend.plot(trend.index.astype(str), trend[col], marker="o", linewidth=2, label=col)
    ax_trend.set_xlabel("Month")
    ax_trend.set_ylabel("Number of Reviews")
    ax_trend.set_title("Monthly Sentiment Trend")
    ax_trend.legend()
    ax_trend.tick_params(axis="x", rotation=45)
    st.pyplot(fig_trend)

st.subheader("🔑 Top Keywords & Word Cloud")

if not keywords_df.empty:
    top10 = keywords_df.head(15)
    colA, colB = st.columns([3, 2])

    with colA:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        ax_bar.barh(top10["keyword"], top10["frequency"])
        ax_bar.invert_yaxis()
        st.pyplot(fig_bar)

    with colB:
        word_freq = dict(zip(keywords_df["keyword"], keywords_df["frequency"]))
        wc = WordCloud(width=400, height=400, background_color="white").generate_from_frequencies(word_freq)
        fig_wc, ax_wc = plt.subplots(figsize=(6, 6))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

st.subheader("📊 Confidence Score Distribution")

if not filtered_df.empty:
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    ax_hist.hist(filtered_df["confidence_score"], bins=25, edgecolor="black")
    ax_hist.set_xlabel("Confidence Score")
    ax_hist.set_ylabel("Count")
    st.pyplot(fig_hist)

with st.expander("📋 Preview Filtered Data"):
    st.dataframe(filtered_df.head(15), use_container_width=True)

st.subheader("💾 Export Options")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    st.download_button(
        "⬇️ Download Filtered Reviews",
        filtered_df.to_csv(index=False).encode("utf-8"),
        "ReviewSense_Filtered_Reviews.csv",
        "text/csv"
    )

with col_dl2:
    if not keywords_df.empty:
        st.download_button(
            "⬇️ Download Keyword List",
            keywords_df.to_csv(index=False).encode("utf-8"),
            "ReviewSense_Keywords.csv",
            "text/csv"
        )

st.success("✅ Dashboard ready! Use the sidebar to explore insights.")