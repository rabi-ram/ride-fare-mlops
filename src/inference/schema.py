from pydantic import BaseModel


class FareRequest(BaseModel):
    passenger_count: int
    trip_distance: float
    pickup_hour: int
    pickup_day_of_week: int
    PULocationID: int
    DOLocationID: int
