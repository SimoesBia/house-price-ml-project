import numpy as np
import pytest

from house_price.config import TARGET_COLUMN
from house_price.data import split_dataset, validate_dataframe
from house_price.exceptions import DataValidationError


def test_validate_accepts_valid_dataframe(synthetic_df):
    validate_dataframe(synthetic_df)


def test_validate_rejects_missing_columns(synthetic_df):
    with pytest.raises(DataValidationError, match="Colunas ausentes"):
        validate_dataframe(synthetic_df.drop(columns=["MedInc"]))


def test_validate_rejects_nulls(synthetic_df):
    broken = synthetic_df.copy()
    broken.loc[0, "HouseAge"] = np.nan
    with pytest.raises(DataValidationError, match="nulos"):
        validate_dataframe(broken)


def test_split_sizes_and_target_not_in_features(synthetic_df):
    X_train, X_test, y_train, y_test = split_dataset(synthetic_df, 0.25, 1)
    assert len(X_test) == len(y_test) == 100
    assert len(X_train) == len(y_train) == 300
    assert TARGET_COLUMN not in X_train.columns
