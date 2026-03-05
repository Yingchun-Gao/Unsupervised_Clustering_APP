import pandas as pd
from pathlib import Path


# File paths for raw and processed datasets
RAW_PATH = "data/raw/mall_customers.csv"
PROCESSED_PATH = "data/processed/mall_customers_clean.csv"


def clean_data(df):
    """
    Removes Customer_ID because it has no meaning for clustering.
    """
    if "Customer_ID" in df.columns:
        df = df.drop(columns=["Customer_ID"])
    return df


def main():
    """
    Load the raw dataset, clean it, and save the processed version.
    """
    df = pd.read_csv(RAW_PATH)

    df_clean = clean_data(df)

    # Create processed folder if it does not exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    df_clean.to_csv(PROCESSED_PATH, index=False)


if __name__ == "__main__":
    main()
