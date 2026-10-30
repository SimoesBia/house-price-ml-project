"""Definição dos modelos candidatos como Pipelines do scikit-learn."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Grade de hiperparâmetros usada com `--tune` (prefixo `model__` = etapa do Pipeline).
RANDOM_FOREST_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 20],
    "model__min_samples_leaf": [1, 3],
}


def build_candidates(random_state: int) -> dict[str, Pipeline]:
    """Modelos comparados no treino. Usar Pipeline evita vazamento de dados (data leakage)."""
    return {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100, random_state=random_state, n_jobs=-1
                    ),
                )
            ]
        ),
    }
