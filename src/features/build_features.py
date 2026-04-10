import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/mall_customers_clean.csv"


def build_features():
    try:
        logger.info("Building features")

        df = pd.read_csv(DATA_PATH)

        features = ["Age", "Annual_Income", "Spending_Score"]

        X = df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return df, X_scaled

    except Exception as e:
        logger.error(f"build_features error: {e}")
        raise
