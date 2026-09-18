from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/" "yellow_tripdata_2025-01.parquet"
)

RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DATA_DIR / "yellow_tripdata_2025-01.parquet"


def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading NYC Taxi dataset...")
    df = pd.read_parquet(DATA_URL)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    download_data()
