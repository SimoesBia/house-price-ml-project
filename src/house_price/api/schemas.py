"""Schemas Pydantic de entrada/saída da API (validação automática + docs OpenAPI)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HouseFeatures(BaseModel):
    """Características de um distrito. Faixas limitadas à região coberta pelo dataset."""

    MedInc: float = Field(..., gt=0, le=20, description="Renda mediana (US$ 10 mil)")
    HouseAge: float = Field(..., ge=0, le=100, description="Idade mediana das casas (anos)")
    AveRooms: float = Field(..., gt=0, le=200, description="Média de cômodos por domicílio")
    AveBedrms: float = Field(..., gt=0, le=50, description="Média de quartos por domicílio")
    Population: float = Field(..., ge=1, le=100_000, description="População do distrito")
    AveOccup: float = Field(..., gt=0, le=2_000, description="Média de moradores por domicílio")
    Latitude: float = Field(..., ge=32, le=42.5, description="Latitude (Califórnia)")
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude (Califórnia)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.98,
                "AveBedrms": 1.02,
                "Population": 322.0,
                "AveOccup": 2.56,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    )


class PredictionResponse(BaseModel):
    predicted_price_usd: float = Field(..., description="Preço mediano previsto, em US$")


class BatchPredictionRequest(BaseModel):
    houses: list[HouseFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions_usd: list[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    metadata: dict[str, Any]
