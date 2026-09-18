import mlflow
import pandas as pd

# Load bundled model inside Docker
MODEL_URI = "models"

model = mlflow.pyfunc.load_model(MODEL_URI)


def predict_fare(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return float(prediction[0])
