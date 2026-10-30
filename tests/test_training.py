import json

from house_price.training import train


def test_train_saves_artifacts_and_picks_best_model(settings, synthetic_df):
    result = train(settings, df=synthetic_df)

    assert result.best_model_name in {"linear_regression", "random_forest"}
    assert settings.model_path.is_file()
    assert (settings.reports_dir / "metrics.json").is_file()
    assert (settings.figures_dir / "feature_importance.png").is_file()

    saved = json.loads((settings.reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["best_model"] == result.best_model_name
    # alvo sintético é quase linear: o R² no teste deve ser alto
    assert result.test_metrics[result.best_model_name]["r2"] > 0.9


def test_train_without_saving_writes_nothing(settings, synthetic_df):
    train(settings, df=synthetic_df, save=False)
    assert not settings.model_path.exists()
