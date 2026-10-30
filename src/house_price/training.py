"""Treino, comparação de modelos e persistência do melhor pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from .config import FEATURE_NAMES, Settings
from .data import load_california_housing, split_dataset
from .evaluation import regression_metrics
from .modeling import RANDOM_FOREST_PARAM_GRID, build_candidates
from .visualization import plot_feature_importance, plot_predictions_vs_actual

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    best_model_name: str
    test_metrics: dict[str, dict[str, float]]
    cv_r2: dict[str, dict[str, float]]
    best_params: dict[str, Any]
    feature_importance: pd.DataFrame
    metadata: dict[str, Any]
    model_path: Path | None = None
    artifacts: list[Path] = field(default_factory=list)


def _tune_random_forest(
    pipeline: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, cv: KFold
) -> dict[str, Any]:
    """GridSearchCV no Random Forest. Retorna os melhores hiperparâmetros."""
    logger.info("Iniciando GridSearchCV (pode levar alguns minutos)...")
    search = GridSearchCV(pipeline, RANDOM_FOREST_PARAM_GRID, cv=cv, scoring="r2", n_jobs=1)
    search.fit(X_train, y_train)
    logger.info(
        "Melhores hiperparâmetros: %s (R² CV=%.4f)", search.best_params_, search.best_score_
    )
    return dict(search.best_params_)


def train(
    settings: Settings,
    df: pd.DataFrame | None = None,
    tune: bool = False,
    save: bool = True,
) -> TrainingResult:
    """Treina os candidatos, escolhe o melhor por R² em validação cruzada e salva o artefato.

    A escolha usa apenas o conjunto de treino (CV); o conjunto de teste fica reservado
    para a avaliação final, evitando otimismo nas métricas reportadas.
    """
    if df is None:
        df = load_california_housing(settings.data_dir)

    X_train, X_test, y_train, y_test = split_dataset(df, settings.test_size, settings.random_state)
    cv = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)
    candidates = build_candidates(settings.random_state)

    rf_params: dict[str, Any] = {}
    if tune:
        rf_params = _tune_random_forest(candidates["random_forest"], X_train, y_train, cv)
        candidates["random_forest"].set_params(**rf_params)

    test_metrics: dict[str, dict[str, float]] = {}
    cv_r2: dict[str, dict[str, float]] = {}
    predictions: dict[str, Any] = {}

    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
        cv_r2[name] = {"mean": float(scores.mean()), "std": float(scores.std())}

        pipeline.fit(X_train, y_train)
        predictions[name] = pipeline.predict(X_test)
        test_metrics[name] = regression_metrics(y_test, predictions[name])
        logger.info(
            "%s | CV R²=%.4f±%.4f | teste: MAE=%.4f RMSE=%.4f R²=%.4f",
            name, cv_r2[name]["mean"], cv_r2[name]["std"],
            test_metrics[name]["mae"], test_metrics[name]["rmse"], test_metrics[name]["r2"],
        )

    best_name = max(cv_r2, key=lambda n: cv_r2[n]["mean"])
    best_model = candidates[best_name]
    logger.info("Melhor modelo (por R² em CV): %s", best_name)

    rf_model = candidates["random_forest"].named_steps["model"]
    importance = (
        pd.DataFrame({"feature": FEATURE_NAMES, "importance": rf_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    metadata: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best_name,
        "feature_names": FEATURE_NAMES,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": settings.random_state,
        "tuned": tune,
        "best_params": rf_params,
        "cv_r2": cv_r2,
        "test_metrics": test_metrics,
        "sklearn_version": sklearn.__version__,
    }

    result = TrainingResult(
        best_model_name=best_name,
        test_metrics=test_metrics,
        cv_r2=cv_r2,
        best_params=rf_params,
        feature_importance=importance,
        metadata=metadata,
    )

    if save:
        _persist(settings, best_model, result, y_test, predictions[best_name])
    return result


def _persist(
    settings: Settings,
    model: BaseEstimator,
    result: TrainingResult,
    y_test: pd.Series,
    best_predictions,
) -> None:
    """Salva modelo (+ metadados), métricas, importâncias e gráficos."""
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "metadata": result.metadata}, settings.model_path)
    result.model_path = settings.model_path

    metrics_path = settings.reports_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(result.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    importance_path = settings.reports_dir / "feature_importance.csv"
    result.feature_importance.to_csv(importance_path, index=False)

    result.artifacts = [
        metrics_path,
        importance_path,
        plot_predictions_vs_actual(y_test, best_predictions, settings.figures_dir),
        plot_feature_importance(result.feature_importance, settings.figures_dir),
    ]
    logger.info("Modelo salvo em %s", settings.model_path)
