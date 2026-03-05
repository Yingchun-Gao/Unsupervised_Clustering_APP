from sklearn.cluster import KMeans
import pickle
import os


def train_kmeans(X_scaled, n_clusters=6):
    """
    Train the KMeans clustering model.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X_scaled)
    return model


def assign_clusters(model, X_scaled):
    """
    Use the trained model to assign cluster labels to the dataset.
    """
    return model.predict(X_scaled)


def save_model(model, path="models/kmeans_model.pkl"):
    """
    Save the trained model to disk so it can be reused later.
    """
    os.makedirs("models", exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)
