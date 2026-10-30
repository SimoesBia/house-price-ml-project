"""Configuração central do projeto, lida de variáveis de ambiente (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_NAMES: list[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_COLUMN = "MedHouseVal"
# No dataset, o alvo está em unidades de US$ 100 mil (ex.: 2.5 == US$ 250.000).
TARGET_SCALE_USD = 100_000


def _load_dotenv(path: Path) -> None:
    """Carrega pares CHAVE=VALOR de um .env sem sobrescrever variáveis já definidas."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    log_level: str
    random_state: int
    test_size: float
    cv_folds: int
    data_dir: Path
    model_path: Path
    reports_dir: Path
    api_host: str
    api_port: int

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            random_state=int(os.getenv("RANDOM_STATE", "42")),
            test_size=float(os.getenv("TEST_SIZE", "0.2")),
            cv_folds=int(os.getenv("CV_FOLDS", "5")),
            data_dir=_resolve(os.getenv("DATA_DIR", "data")),
            model_path=_resolve(os.getenv("MODEL_PATH", "models/house_price_model.joblib")),
            reports_dir=_resolve(os.getenv("REPORTS_DIR", "reports")),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
