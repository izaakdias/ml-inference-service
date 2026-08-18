# ML Inference Service

A versioned, production-shaped inference API for mobility demand scoring.

The service exposes a stable prediction contract around a deterministic baseline model. It is intentionally self-contained and uses synthetic features so it can be published safely without copying any proprietary Leaf.app implementation.

## What this demonstrates

- Versioned model metadata and prediction contracts
- FastAPI inference endpoints with OpenAPI documentation
- Input validation and bounded output scores
- Health and readiness checks
- In-memory latency and request metrics
- Docker packaging, unit tests, and CI

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
uvicorn ml_inference_service.api:app --reload
```

Try the service:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/model-info
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"requests_last_hour": 28, "active_drivers": 12, "avg_pickup_eta_minutes": 9.5, "rainfall_mm": 4.0, "hour": 18}'
```

## API contract

`POST /v1/predict` returns:

- `model_name` and `model_version`;
- a bounded `demand_score`;
- a demand `class`;
- a confidence estimate;
- feature contributions for basic explainability;
- request latency in milliseconds.

The model is a transparent baseline intended to demonstrate serving and operational interfaces. A production implementation would load an approved artifact from a registry and validate it against a model contract.

## Testing

```bash
pytest
```

## Next iterations

1. Load a serialized model artifact through a registry adapter.
2. Add authentication and request correlation IDs.
3. Export Prometheus-compatible metrics.
4. Add canary model routing and rollback checks.
