"""A transparent baseline model used to demonstrate inference contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp


MODEL_NAME = "mobility-demand-baseline"
MODEL_VERSION = "1.0.0"


@dataclass(frozen=True)
class Prediction:
    score: float
    demand_class: str
    confidence: float
    contributions: dict[str, float]


def _sigmoid(value: float) -> float:
    return 1 / (1 + exp(-value))


def predict(
    requests_last_hour: float,
    active_drivers: float,
    avg_pickup_eta_minutes: float,
    rainfall_mm: float,
    hour: int,
) -> Prediction:
    request_signal = min(requests_last_hour / 40, 1.0)
    supply_signal = 1 - min(active_drivers / 40, 1.0)
    eta_signal = min(max(avg_pickup_eta_minutes - 4, 0) / 12, 1.0)
    weather_signal = min(rainfall_mm / 12, 1.0)
    peak_signal = float(hour in {7, 8, 9, 17, 18, 19, 20})
    contributions = {
        "request_pressure": round(0.38 * request_signal, 4),
        "supply_pressure": round(0.24 * supply_signal, 4),
        "eta_pressure": round(0.18 * eta_signal, 4),
        "weather_pressure": round(0.10 * weather_signal, 4),
        "peak_hour": round(0.10 * peak_signal, 4),
    }
    raw_score = sum(contributions.values())
    score = round(_sigmoid((raw_score - 0.45) * 8), 4)
    demand_class = "critical" if score >= 0.8 else "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"
    confidence = round(min(0.99, 0.55 + abs(score - 0.5) * 0.75), 4)
    return Prediction(score, demand_class, confidence, contributions)
