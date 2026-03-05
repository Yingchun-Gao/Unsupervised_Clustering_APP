import pandas as pd
from sklearn.preprocessing import StandardScaler


# Location of cleaned dataset
DATA_PATH = "data/processed/mall_customers_clean.csv"


def build_features():
    """
    Load the cleaned dataset and prepare features for clustering.
    Selected features: Age, Annual_Income, Spending_Score
    Features are standardized because KMeans uses distance calculations.
    """
    df = pd.read_csv(DATA_PATH)

    features = ["Age", "Annual_Income", "Spending_Score"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled
