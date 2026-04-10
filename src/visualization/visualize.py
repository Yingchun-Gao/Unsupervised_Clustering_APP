import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("models", exist_ok=True)


def plot_elbow(k_values, wcss):
    try:
        plt.figure(figsize=(6, 4))

        plt.plot(k_values, wcss, marker="o")
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS")
        plt.title("Elbow Method")

        plt.tight_layout()
        plt.savefig("models/elbow_method.png")
        plt.close()

        logger.info("Elbow plot saved")

    except Exception as e:
        logger.error(f"plot_elbow error: {e}")
        raise


def plot_silhouette(k_values, scores):
    try:
        plt.figure(figsize=(6, 4))

        plt.plot(k_values, scores, marker="o")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis")

        plt.tight_layout()
        plt.savefig("models/silhouette_scores.png")
        plt.close()

        logger.info("Silhouette plot saved")

    except Exception as e:
        logger.error(f"plot_silhouette error: {e}")
        raise


def plot_clusters(df):
    try:
        plt.figure(figsize=(6, 4))

        plt.scatter(df["Annual_Income"], df["Spending_Score"], c=df["Cluster"])
        plt.xlabel("Annual Income")
        plt.ylabel("Spending Score")
        plt.title("Customer Segments")

        plt.tight_layout()
        plt.savefig("models/customer_clusters.png")
        plt.close()

        logger.info("Cluster plot saved")

    except Exception as e:
        logger.error(f"plot_clusters error: {e}")
        raise
