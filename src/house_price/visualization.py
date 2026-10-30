"""Geração de gráficos (salvos em disco, sem `plt.show()` bloqueante)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela: funciona em servidores, Docker e CI

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import TARGET_COLUMN  # noqa: E402

logger = logging.getLogger(__name__)

PRICE_LABEL = "Preço mediano (US$ 100 mil)"


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_price_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], bins=30, ax=ax)
    ax.set(title="Distribuição dos preços das casas", xlabel=PRICE_LABEL, ylabel="Distritos")
    return _save(fig, out_dir, "price_distribution.png")


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlação entre variáveis")
    return _save(fig, out_dir, "correlation_heatmap.png")


def plot_income_vs_price(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="MedInc", y=TARGET_COLUMN, alpha=0.3, s=8, ax=ax)
    ax.set(
        title="Renda média vs preço das casas",
        xlabel="Renda média da região",
        ylabel=PRICE_LABEL,
    )
    return _save(fig, out_dir, "income_vs_price.png")


def plot_predictions_vs_actual(y_true, y_pred, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, s=8, ax=ax)
    lim = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    ax.plot(lim, lim, "r--", linewidth=1, label="Previsão perfeita")
    ax.set(title="Previsão vs valor real (teste)", xlabel="Preço real", ylabel="Preço previsto")
    ax.legend()
    return _save(fig, out_dir, "predictions_vs_actual.png")


def plot_feature_importance(importance: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance, x="importance", y="feature", ax=ax)
    ax.set(
        title="Importância das variáveis (Random Forest)",
        xlabel="Importância",
        ylabel="Variável",
    )
    return _save(fig, out_dir, "feature_importance.png")


def generate_eda_report(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Gera os gráficos da análise exploratória."""
    paths = [
        plot_price_distribution(df, out_dir),
        plot_correlation(df, out_dir),
        plot_income_vs_price(df, out_dir),
    ]
    logger.info("EDA: %d gráficos salvos em %s", len(paths), out_dir)
    return paths
