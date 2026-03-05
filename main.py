from src.data.make_dataset import main as make_dataset
from src.features.build_features import build_features
from src.models.train_model import train_kmeans, assign_clusters, save_model
from src.visualization.visualize import plot_elbow, plot_silhouette, plot_clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_pipeline():

    # Step 1: Clean dataset
    make_dataset()

    # Step 2: Build features
    df, X_scaled = build_features()

    # Step 3: Evaluate different cluster numbers
    k_values = list(range(3, 9))
    wcss = []
    silhouette_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)

        wcss.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # Step 4: Visualize evaluation metrics
    plot_elbow(k_values, wcss)
    plot_silhouette(k_values, silhouette_scores)

    # Step 5: Train final model (k=6 based on analysis)
    final_model = train_kmeans(X_scaled, n_clusters=6)

    # Step 6: Assign clusters
    df["Cluster"] = assign_clusters(final_model, X_scaled)

    # Step 7: Save trained model
    save_model(final_model)

    # Step 8: Visualize final clusters
    plot_clusters(df)


if __name__ == "__main__":
    run_pipeline()
