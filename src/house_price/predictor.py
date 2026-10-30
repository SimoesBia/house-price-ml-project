"""Carregamento do modelo treinado e inferência."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import FEATURE_NAMES, TARGET_SCALE_USD
from .exceptions import ModelLoadError, ModelNotFoundError, PredictionError

logger = logging.getLogger(__name__)


class HousePricePredictor:
    """Encapsula o pipeline treinado e seus metadados."""

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self._model = model
        self.metadata = metadata

    @classmethod
    def load(cls, path: Path) -> HousePricePredictor:
        if not path.is_file():
            raise ModelNotFoundError(
                f"Modelo não encontrado em '{path}'. Execute `python main.py train` primeiro."
            )
        try:
            # Atenção: joblib usa pickle — carregue apenas arquivos gerados por você.
            bundle = joblib.load(path)
            return cls(bundle["model"], bundle.get("metadata", {}))
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Falha ao ler o modelo em '{path}': {exc}") from exc

    def predict(self, records: list[dict[str, float]]) -> list[float]:
        """Prevê o preço em dólares para cada registro."""
        if not records:
            raise PredictionError("Nenhum registro enviado para previsão.")
        try:
            frame = pd.DataFrame.from_records(records)[FEATURE_NAMES]
        except KeyError as exc:
            raise PredictionError(f"Features ausentes: {exc}. Esperado: {FEATURE_NAMES}") from exc

        raw = np.clip(self._model.predict(frame), 0, None)  # preço não pode ser negativo
        return [round(float(value) * TARGET_SCALE_USD, 2) for value in raw]
