"""Fixtures compartilhadas: dataset sintético (sem rede) e settings em diretório temporário."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from house_price.config import FEATURE_NAMES, TARGET_COLUMN, Settings
from house_price.predictor import HousePricePredictor
from house_price.training import train


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "MedInc": rng.uniform(1, 10, n),
            "HouseAge": rng.uniform(1, 52, n),
            "AveRooms": rng.uniform(3, 8, n),
            "AveBedrms": rng.uniform(0.8, 1.5, n),
            "Population": rng.uniform(100, 3000, n),
            "AveOccup": rng.uniform(1.5, 4, n),
            "Latitude": rng.uniform(32.5, 42, n),
            "Longitude": rng.uniform(-124, -114.5, n),
        }
    )
    df[TARGET_COLUMN] = 0.4 * df["MedInc"] + 0.01 * df["HouseAge"] + rng.normal(0, 0.1, n)
    return df[[*FEATURE_NAMES, TARGET_COLUMN]]


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        log_level="WARNING",
        random_state=42,
        test_size=0.2,
        cv_folds=3,
        data_dir=tmp_path / "data",
        model_path=tmp_path / "models" / "model.joblib",
        reports_dir=tmp_path / "reports",
        api_host="127.0.0.1",
        api_port=8000,
    )


@pytest.fixture()
def trained_predictor(settings, synthetic_df) -> HousePricePredictor:
    train(settings, df=synthetic_df)
    return HousePricePredictor.load(settings.model_path)


@pytest.fixture()
def valid_house() -> dict[str, float]:
    return {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.98,
        "AveBedrms": 1.02,
        "Population": 322.0,
        "AveOccup": 2.56,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
