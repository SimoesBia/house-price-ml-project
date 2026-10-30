"""Aplicação FastAPI: endpoints de saúde, metadados do modelo e previsão."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from .. import __version__
from ..config import Settings, get_settings
from ..exceptions import ModelLoadError, ModelNotFoundError, PredictionError
from ..logging_config import setup_logging
from ..predictor import HousePricePredictor
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    HouseFeatures,
    ModelInfoResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


def create_app(
    predictor: HousePricePredictor | None = None, settings: Settings | None = None
) -> FastAPI:
    """Fábrica da aplicação. Aceitar `predictor` por parâmetro facilita testes."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        app.state.predictor = predictor
        if app.state.predictor is None:
            try:
                app.state.predictor = HousePricePredictor.load(settings.model_path)
                logger.info("Modelo carregado de %s", settings.model_path)
            except (ModelNotFoundError, ModelLoadError) as exc:
                logger.warning("API iniciada sem modelo: %s", exc)
        yield

    app = FastAPI(
        title="House Price Prediction API",
        description="Previsão do preço mediano de imóveis na Califórnia com Machine Learning.",
        version=__version__,
        lifespan=lifespan,
    )

    def get_predictor(request: Request) -> HousePricePredictor:
        current = request.app.state.predictor
        if current is None:
            raise HTTPException(
                status_code=503, detail="Modelo indisponível. Execute o treino (`make train`)."
            )
        return current

    @app.get("/health", response_model=HealthResponse, tags=["Operações"])
    def health(request: Request) -> HealthResponse:
        loaded = request.app.state.predictor is not None
        return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)

    @app.get("/api/v1/model", response_model=ModelInfoResponse, tags=["Modelo"])
    def model_info(model: HousePricePredictor = Depends(get_predictor)) -> ModelInfoResponse:
        return ModelInfoResponse(metadata=model.metadata)

    @app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Previsão"])
    def predict(
        house: HouseFeatures, model: HousePricePredictor = Depends(get_predictor)
    ) -> PredictionResponse:
        try:
            (price,) = model.predict([house.model_dump()])
        except PredictionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse(predicted_price_usd=price)

    @app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Previsão"])
    def predict_batch(
        payload: BatchPredictionRequest, model: HousePricePredictor = Depends(get_predictor)
    ) -> BatchPredictionResponse:
        try:
            prices = model.predict([house.model_dump() for house in payload.houses])
        except PredictionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return BatchPredictionResponse(count=len(prices), predictions_usd=prices)

    return app


app = create_app()
