from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/yellow_tripdata_2025-01.parquet")
PROCESSED_PATH = Path("data/processed/taxi_features.parquet")


def build_features():

    print("Loading raw dataset...")
    df = pd.read_parquet(RAW_PATH)

    original_rows = len(df)

    # -----------------------
    # Data Cleaning
    # -----------------------

    df = df[df["passenger_count"].between(1, 6)]

    df = df[df["trip_distance"] > 0]

    df = df[df["fare_amount"] > 0]

    # -----------------------
    # Feature Engineering
    # -----------------------

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

    df["pickup_day_of_week"] = (
        df["tpep_pickup_datetime"].dt.dayofweek
    )

    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] -
        df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[df["trip_duration_min"] > 0]

    # Save
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)

    print("\nFeature engineering completed")
    print(f"Original rows : {original_rows:,}")
    print(f"Processed rows: {len(df):,}")
    print(f"Removed rows  : {original_rows - len(df):,}")

    print(f"\nSaved to: {PROCESSED_PATH}")


if __name__ == "__main__":
    build_features()

    