import pytest

from house_price.exceptions import ModelNotFoundError, PredictionError
from house_price.predictor import HousePricePredictor


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        HousePricePredictor.load(tmp_path / "nao_existe.joblib")


def test_predict_returns_usd_values(trained_predictor, valid_house):
    (price,) = trained_predictor.predict([valid_house])
    assert price > 10_000  # em dólares, não em unidades de US$ 100 mil


def test_predict_batch_keeps_order_and_size(trained_predictor, valid_house):
    low = {**valid_house, "MedInc": 1.5}
    prices = trained_predictor.predict([low, valid_house])
    assert len(prices) == 2 and prices[0] < prices[1]


def test_predict_missing_feature_raises(trained_predictor, valid_house):
    incomplete = {k: v for k, v in valid_house.items() if k != "MedInc"}
    with pytest.raises(PredictionError):
        trained_predictor.predict([incomplete])


def test_predict_empty_raises(trained_predictor):
    with pytest.raises(PredictionError):
        trained_predictor.predict([])
