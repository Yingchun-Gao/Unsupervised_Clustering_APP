import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/mall_customers.csv"
PROCESSED_PATH = "data/processed/mall_customers_clean.csv"


def clean_data(df):
    try:
        if "Customer_ID" in df.columns:
            df = df.drop(columns=["Customer_ID"])
        return df

    except Exception as e:
        logger.error(f"clean_data error: {e}")
        raise


def main():
    try:
        logger.info("Loading raw data")

        df = pd.read_csv(RAW_PATH)

        df_clean = clean_data(df)

        Path("data/processed").mkdir(parents=True, exist_ok=True)

        df_clean.to_csv(PROCESSED_PATH, index=False)
        logger.info("Cleaned dataset saved")

    except Exception as e:
        logger.error(f"make_dataset error: {e}")
        raise


if __name__ == "__main__":
    main()
