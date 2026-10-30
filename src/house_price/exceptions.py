"""Exceções específicas do domínio, para tratamento de erros explícito."""


class HousePriceError(Exception):
    """Exceção base do projeto."""


class DataLoadError(HousePriceError):
    """Falha ao carregar o dataset (ex.: sem internet no primeiro download)."""


class DataValidationError(HousePriceError):
    """O dataset não tem o formato esperado."""


class ModelNotFoundError(HousePriceError):
    """Arquivo do modelo treinado não encontrado."""


class ModelLoadError(HousePriceError):
    """O arquivo do modelo existe, mas não pôde ser lido."""


class PredictionError(HousePriceError):
    """Entrada inválida para previsão."""
