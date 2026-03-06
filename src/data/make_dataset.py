import pandas as pd
from pathlib import Path


RAW_PATH = "data/raw/mall_customers.csv"
PROCESSED_PATH = "data/processed/mall_customers_clean.csv"


def clean_data(df):

    if "Customer_ID" in df.columns:
        df = df.drop(columns=["Customer_ID"])
    return df


def main():

    df = pd.read_csv(RAW_PATH)

    df_clean = clean_data(df)

    # Create processed folder if it does not exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    df_clean.to_csv(PROCESSED_PATH, index=False)


if __name__ == "__main__":
    main()
