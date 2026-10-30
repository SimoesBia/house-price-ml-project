"""Carregamento, validação e divisão do dataset California Housing."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .config import FEATURE_NAMES, TARGET_COLUMN
from .exceptions import DataLoadError, DataValidationError

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> None:
    """Garante colunas esperadas, ausência de nulos e dataset não vazio."""
    missing = [c for c in [*FEATURE_NAMES, TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colunas ausentes no dataset: {missing}")
    if df.empty:
        raise DataValidationError("O dataset está vazio.")
    null_count = int(df[[*FEATURE_NAMES, TARGET_COLUMN]].isna().sum().sum())
    if null_count:
        raise DataValidationError(f"O dataset contém {null_count} valores nulos.")


def load_california_housing(data_dir: Path) -> pd.DataFrame:
    """Carrega o dataset, usando cache local em `data_dir` (download só na 1ª vez)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        frame = fetch_california_housing(data_home=str(data_dir), as_frame=True).frame
    except Exception as exc:  # noqa: BLE001 - falhas de rede/IO variam
        raise DataLoadError(
            "Não foi possível carregar o California Housing. "
            "Verifique a conexão na primeira execução (depois o cache local é usado)."
        ) from exc

    validate_dataframe(frame)
    logger.info("Dataset carregado: %d linhas, %d colunas.", *frame.shape)
    return frame


def split_dataset(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa X/y e divide em treino e teste."""
    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
