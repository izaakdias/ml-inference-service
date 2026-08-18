"""Request and response schemas."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    requests_last_hour: float = Field(ge=0, le=10_000)
    active_drivers: float = Field(ge=0, le=10_000)
    avg_pickup_eta_minutes: float = Field(ge=0, le=240)
    rainfall_mm: float = Field(ge=0, le=500)
    hour: int = Field(ge=0, le=23)


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    demand_score: float
    demand_class: str
    confidence: float
    feature_contributions: dict[str, float]
    latency_ms: float


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    input_schema: list[str]
    status: str
