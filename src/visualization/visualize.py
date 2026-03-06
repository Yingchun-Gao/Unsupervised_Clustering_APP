import matplotlib.pyplot as plt
import os

# ensure models folder exists
os.makedirs("models", exist_ok=True)


def plot_elbow(k_values, wcss):

    plt.figure(figsize=(6, 4))

    plt.plot(k_values, wcss, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")

    plt.tight_layout()
    plt.savefig("models/elbow_method.png")

    plt.close()


def plot_silhouette(k_values, scores):

    plt.figure(figsize=(6, 4))

    plt.plot(k_values, scores, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")

    plt.tight_layout()
    plt.savefig("models/silhouette_scores.png")

    plt.close()


def plot_clusters(df):

    plt.figure(figsize=(6, 4))

    plt.scatter(df["Annual_Income"], df["Spending_Score"], c=df["Cluster"])
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.title("Customer Segments")

    plt.tight_layout()
    plt.savefig("models/customer_clusters.png")

    plt.close()
