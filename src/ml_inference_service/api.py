"""FastAPI application for versioned demand inference."""

from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI

from .model import MODEL_NAME, MODEL_VERSION, predict
from .schemas import ModelInfo, PredictionRequest, PredictionResponse

app = FastAPI(
    title="ML Inference Service",
    version="0.1.0",
    description="Versioned and explainable demand scoring API.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/v1/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    return ModelInfo(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        input_schema=[
            "requests_last_hour",
            "active_drivers",
            "avg_pickup_eta_minutes",
            "rainfall_mm",
            "hour",
        ],
        status="loaded",
    )


@app.post("/v1/predict", response_model=PredictionResponse)
def inference(request: PredictionRequest) -> PredictionResponse:
    started = perf_counter()
    result = predict(**request.model_dump())
    latency_ms = round((perf_counter() - started) * 1000, 3)
    return PredictionResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        demand_score=result.score,
        demand_class=result.demand_class,
        confidence=result.confidence,
        feature_contributions=result.contributions,
        latency_ms=latency_ms,
    )
