import math

from house_price.evaluation import regression_metrics


def test_perfect_prediction():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics["mae"] == 0 and metrics["mse"] == 0 and metrics["r2"] == 1


def test_rmse_is_sqrt_of_mse():
    metrics = regression_metrics([0.0, 0.0], [1.0, 3.0])
    assert math.isclose(metrics["rmse"], math.sqrt(metrics["mse"]))
