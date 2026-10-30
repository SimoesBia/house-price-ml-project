"""Ponto de entrada (CLI): eda | train | predict | serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from house_price.config import FEATURE_NAMES, get_settings  # noqa: E402
from house_price.exceptions import HousePriceError  # noqa: E402
from house_price.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("house_price.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="House Price ML — pipeline e API")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("eda", help="Gera os gráficos da análise exploratória")

    train_cmd = sub.add_parser("train", help="Treina, avalia e salva o melhor modelo")
    train_cmd.add_argument(
        "--tune", action="store_true", help="Executa GridSearchCV no Random Forest"
    )

    predict_cmd = sub.add_parser("predict", help="Faz uma previsão pela linha de comando")
    for name in FEATURE_NAMES:
        predict_cmd.add_argument(f"--{name}", type=float, required=True)

    sub.add_parser("serve", help="Sobe a API REST (FastAPI + Uvicorn)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    try:
        if args.command == "eda":
            from house_price.data import load_california_housing
            from house_price.visualization import generate_eda_report

            generate_eda_report(load_california_housing(settings.data_dir), settings.figures_dir)

        elif args.command == "train":
            from house_price.training import train

            result = train(settings, tune=args.tune)
            print(json.dumps(result.test_metrics, indent=2))
            print(f"Melhor modelo: {result.best_model_name}")

        elif args.command == "predict":
            from house_price.predictor import HousePricePredictor

            predictor = HousePricePredictor.load(settings.model_path)
            features = {name: getattr(args, name) for name in FEATURE_NAMES}
            (price,) = predictor.predict([features])
            print(f"Preço previsto: US$ {price:,.2f}")

        elif args.command == "serve":
            import uvicorn

            uvicorn.run("house_price.api.app:app", host=settings.api_host, port=settings.api_port)

    except HousePriceError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
