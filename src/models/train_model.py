from sklearn.cluster import KMeans
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_kmeans(X_scaled, n_clusters=6):
    try:
        logger.info(f"Training KMeans with k={n_clusters}")

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        model.fit(X_scaled)

        return model

    except Exception as e:
        logger.error(f"train_kmeans error: {e}")
        raise


def assign_clusters(model, X_scaled):
    try:
        return model.predict(X_scaled)

    except Exception as e:
        logger.error(f"assign_clusters error: {e}")
        raise


def save_model(model, path="models/kmeans_model.pkl"):
    try:
        os.makedirs("models", exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(model, f)

        logger.info("Model saved")

    except Exception as e:
        logger.error(f"save_model error: {e}")
        raise
