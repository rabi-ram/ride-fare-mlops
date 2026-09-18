from pathlib import Path
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_PATH = Path("data/processed/taxi_features.parquet")

# Environment configuration
SAMPLE_SIZE = os.getenv("SAMPLE_SIZE")


# -------------------------------------------------
# Load dataset only once
# -------------------------------------------------
def load_data(sample_size=None):

    if sample_size:
        print(f"Loading dataset (sample={sample_size:,})...")
    else:
        print("Loading full production dataset...")

    df = pd.read_parquet(DATA_PATH)

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    features = [
        "passenger_count",
        "trip_distance",
        "pickup_hour",
        "pickup_day_of_week",
        "PULocationID",
        "DOLocationID",
    ]

    X = df[features]
    y = df["fare_amount"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


# -------------------------------------------------
# Preprocessing pipeline
# -------------------------------------------------
def build_pipeline(model):

    numeric = [
        "passenger_count",
        "trip_distance",
        "pickup_hour",
        "pickup_day_of_week",
    ]

    categorical = [
        "PULocationID",
        "DOLocationID",
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# -------------------------------------------------
# Generic trainer
# -------------------------------------------------
def train_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

    print(f"\nTraining {name}...")

    pipeline = build_pipeline(model)

    with mlflow.start_run(run_name=name):

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        rmse = root_mean_squared_error(y_test, predictions)

        mlflow.log_param("model", name)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
        )

    print(f"{name} RMSE: {rmse:.2f}")


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":

    mlflow.set_experiment("ride-fare-mlops")

    sample_size = int(SAMPLE_SIZE) if SAMPLE_SIZE else None

    X_train, X_test, y_train, y_test = load_data(sample_size)

    train_model(
        "LinearRegression",
        LinearRegression(),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    train_model(
        "RandomForest",
        RandomForestRegressor(
            n_estimators=20,
            max_depth=15,
            n_jobs=-1,
            random_state=42,
        ),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    train_model(
        "XGBoost",
        XGBRegressor(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        ),
        X_train,
        X_test,
        y_train,
        y_test,
    )
