from sklearn.cluster import KMeans
import pickle
import os


def train_kmeans(X_scaled, n_clusters=6):

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled)
    return model


def assign_clusters(model, X_scaled):

    return model.predict(X_scaled)


def save_model(model, path="models/kmeans_model.pkl"):

    os.makedirs("models", exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)
