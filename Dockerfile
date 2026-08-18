FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
COPY src ./src

EXPOSE 8000
CMD ["uvicorn", "ml_inference_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
