import matplotlib.pyplot as plt


def plot_elbow(k_values, wcss):
    """
    Plot the Elbow Method graph to help datermine the optimal number of clusters.
    """
    plt.plot(k_values, wcss, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.show()


def plot_silhouette(k_values, scores):
    """
    Plot silhouette scores for different numbers of clusters.
    Higher scores indicate better clustering separation.
    """
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.show()


def plot_clusters(df):
    """
    Scatter plot showing the resulting customer clusters.
    """
    plt.scatter(df["Annual_Income"], df["Spending_Score"], c=df["Cluster"])
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.title("Customer Segments")
    plt.show()
