from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.inference.predict import predict_fare
from src.inference.schema import FareRequest

app = FastAPI(
    title="Ride Fare Prediction API",
    version="1.0",
)

# ADD THIS LINE
Instrumentator().instrument(app).expose(app)


@app.get("/")
def home():
    return {"message": "Ride Fare MLOps API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: FareRequest):
    fare = predict_fare(request.model_dump())
    return {"predicted_fare": round(fare, 2)}
