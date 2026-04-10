from src.data.make_dataset import main as make_dataset
from src.features.build_features import build_features
from src.models.train_model import train_kmeans, assign_clusters, save_model
from src.visualization.visualize import plot_elbow, plot_silhouette, plot_clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline():
    try:
        logger.info("Pipeline started")

        make_dataset()

        df, X_scaled = build_features()

        k_values = list(range(3, 9))
        wcss = []
        silhouette_scores = []

        for k in k_values:
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X_scaled)

            wcss.append(model.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))

        plot_elbow(k_values, wcss)
        plot_silhouette(k_values, silhouette_scores)

        final_model = train_kmeans(X_scaled, n_clusters=6)

        df["Cluster"] = assign_clusters(final_model, X_scaled)

        save_model(final_model)

        plot_clusters(df)

        logger.info("Pipeline finished")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
