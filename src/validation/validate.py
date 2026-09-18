from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/yellow_tripdata_2025-01.parquet")
EXPECTED_COLUMNS = 20


def validate_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")

    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Schema
    assert (
        len(df.columns) == EXPECTED_COLUMNS
    ), f"Expected {EXPECTED_COLUMNS} columns, found {len(df.columns)}"

    # Null check
    pickup_nulls = df["tpep_pickup_datetime"].isna().sum()
    assert pickup_nulls == 0, f"Pickup datetime has {pickup_nulls} null values"

    # Passenger count report
    invalid_passengers = df[
        ~df["passenger_count"].isna() & ~df["passenger_count"].between(1, 6)
    ]

    print(f"Invalid passenger rows: {len(invalid_passengers):,}")

    if len(invalid_passengers) > 0:
        print("\nPassenger count distribution:")
        print(df["passenger_count"].value_counts().sort_index())
        raise ValueError("Passenger count validation failed")

    # Distance
    negative_distance = (df["trip_distance"] < 0).sum()
    assert (
        negative_distance == 0
    ), f"{negative_distance} rows have negative trip distance"

    # Fare
    invalid_fare = (df["fare_amount"] <= 0).sum()
    assert invalid_fare == 0, f"{invalid_fare} rows have invalid fare"

    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates:,}")

    print("\nAll validation checks passed!")


if __name__ == "__main__":
    validate_dataset()
