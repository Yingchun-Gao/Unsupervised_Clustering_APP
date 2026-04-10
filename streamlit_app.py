import streamlit as st
import matplotlib.pyplot as plt
from src.features.build_features import build_features
from src.models.train_model import train_kmeans, assign_clusters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Mall Customer Segmentation")

st.title("Mall Customer Segmentation")

st.write(
    "This app segments mall customers using KMeans clustering "
    "based on Age, Annual Income, and Spending Score."
)

try:
    df, X_scaled = build_features()
except Exception as e:
    logger.error(f"Data loading error: {e}")
    st.error("Failed to load data")
    st.stop()

st.sidebar.header("Model Settings")

k = st.sidebar.slider("Number of clusters", 2, 10, 6)

try:
    model = train_kmeans(X_scaled, n_clusters=k)
    df["Cluster"] = assign_clusters(model, X_scaled)
except Exception as e:
    logger.error(f"Model error: {e}")
    st.error("Model training failed")
    st.stop()

st.subheader("Customer Clusters")

fig, ax = plt.subplots()

ax.scatter(df["Annual_Income"], df["Spending_Score"], c=df["Cluster"])

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments")

st.pyplot(fig)

st.subheader("Clustered Dataset")
st.dataframe(df)
