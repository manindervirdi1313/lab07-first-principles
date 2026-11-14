"""Data splitting and preprocessing for bike rental prediction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bikes_data import TARGET, make_rentals

NUMERIC_FEATURES = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["station_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class PreparedData:
    """Hold prepared training, validation, and test data."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer


def split_data(
    frame: pd.DataFrame,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into 60% training, 20% validation, and 20% test."""
    train_frame, remaining = train_test_split(
        frame,
        test_size=0.40,
        random_state=seed,
    )
    val_frame, test_frame = train_test_split(
        remaining,
        test_size=0.50,
        random_state=seed,
    )
    return train_frame, val_frame, test_frame


def prepare_data(
    frame: pd.DataFrame | None = None,
    seed: int = 2026,
) -> PreparedData:
    """Split data and fit preprocessing using training rows only."""
    if frame is None:
        frame = make_rentals()

    train_frame, val_frame, test_frame = split_data(frame, seed)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(train_frame[FEATURES]).astype(np.float32)
    X_val = preprocessor.transform(val_frame[FEATURES]).astype(np.float32)
    X_test = preprocessor.transform(test_frame[FEATURES]).astype(np.float32)

    y_train = train_frame[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val = val_frame[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test = test_frame[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)

    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        preprocessor=preprocessor,
    )


if __name__ == "__main__":
    data = prepare_data()
    print("Training:", data.X_train.shape, data.y_train.shape)
    print("Validation:", data.X_val.shape, data.y_val.shape)
    print("Testing:", data.X_test.shape, data.y_test.shape)
