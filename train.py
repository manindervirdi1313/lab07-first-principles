"""Training utilities for the bike rental neural network."""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from model import BikeNet, make_loader
from prepare import prepare_data


def set_seed(seed: int = 2026) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
) -> tuple[float, float]:
    """Calculate average loss and mean absolute error."""
    model.eval()
    total_loss = 0.0
    total_absolute_error = 0.0
    total_rows = 0

    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features)
            loss = loss_function(predictions, targets)

            batch_size = len(features)
            total_loss += loss.item() * batch_size
            total_absolute_error += torch.abs(predictions - targets).sum().item()
            total_rows += batch_size

    return (
        total_loss / total_rows,
        total_absolute_error / total_rows,
    )


def train_model(
    epochs: int = 100,
    learning_rate: float = 0.001,
    hidden_size: int = 64,
    seed: int = 2026,
) -> tuple[BikeNet, list[dict[str, float]], int]:
    """Train the network and restore its best validation weights."""
    set_seed(seed)
    data = prepare_data(seed=seed)

    train_loader = make_loader(
        data.X_train,
        data.y_train,
        batch_size=64,
        shuffle=True,
        seed=seed,
    )
    val_loader = make_loader(
        data.X_val,
        data.y_val,
        batch_size=64,
        shuffle=False,
    )

    model = BikeNet(
        input_size=data.X_train.shape[1],
        hidden_size=hidden_size,
    )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_rows = 0

        for features, targets in train_loader:
            optimizer.zero_grad()

            predictions = model(features)
            loss = loss_function(predictions, targets)

            loss.backward()
            optimizer.step()

            batch_size = len(features)
            total_train_loss += loss.item() * batch_size
            total_train_rows += batch_size

        train_loss = total_train_loss / total_train_rows
        val_loss, val_mae = evaluate(
            model,
            val_loader,
            loss_function,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Validation loss: {val_loss:.4f} | "
                f"Validation MAE: {val_mae:.4f}"
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, best_epoch


if __name__ == "__main__":
    trained_model, training_history, best_epoch = train_model()
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Save training history
    history_frame = pd.DataFrame(training_history)
    history_frame.to_csv(reports_dir / "history.csv", index=False)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(
        history_frame["epoch"],
        history_frame["train_loss"],
        label="Training Loss",
    )
    plt.plot(
        history_frame["epoch"],
        history_frame["val_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error Loss")
    plt.title("Training and Validation Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_dir / "learning_curve.png", dpi=150)
    plt.close()

    print("Training history saved to reports/history.csv")
    print("Learning curve saved to reports/learning_curve.png")
    # Run two one-variable-at-a-time experiments.
    baseline_best = min(
        training_history,
        key=lambda item: item["val_loss"],
    )

    experiment_results = [
        {
            "experiment": "baseline",
            "learning_rate": 0.001,
            "hidden_size": 64,
            "best_epoch": int(baseline_best["epoch"]),
            "val_loss": baseline_best["val_loss"],
            "val_mae": baseline_best["val_mae"],
        }
    ]

    # Experiment 1: change learning rate only.
    _, lr_history, lr_best_epoch = train_model(
        learning_rate=0.0005,
        hidden_size=64,
    )

    lr_best = min(lr_history, key=lambda item: item["val_loss"])

    experiment_results.append(
        {
            "experiment": "lower_learning_rate",
            "learning_rate": 0.0005,
            "hidden_size": 64,
            "best_epoch": lr_best_epoch,
            "val_loss": lr_best["val_loss"],
            "val_mae": lr_best["val_mae"],
        }
    )

    # Experiment 2: change hidden size only.
    _, size_history, size_best_epoch = train_model(
        learning_rate=0.001,
        hidden_size=128,
    )

    size_best = min(size_history, key=lambda item: item["val_loss"])

    experiment_results.append(
        {
            "experiment": "larger_hidden_size",
            "learning_rate": 0.001,
            "hidden_size": 128,
            "best_epoch": size_best_epoch,
            "val_loss": size_best["val_loss"],
            "val_mae": size_best["val_mae"],
        }
    )

    # Save experiment results.
    experiments_frame = pd.DataFrame(experiment_results)
    experiments_frame.to_csv(
        reports_dir / "experiments.csv",
        index=False,
    )

    print("\nExperiment comparison:")
    print(experiments_frame.to_string(index=False))
    print("\nExperiment results saved to reports/experiments.csv")

    # Recreate the same deterministic data split.
    data = prepare_data()

    # Evaluate neural network on the test set.
    trained_model.eval()
    test_features = torch.tensor(data.X_test, dtype=torch.float32)

    with torch.no_grad():
        neural_predictions = trained_model(test_features).numpy().ravel()

    actual = data.y_test.ravel()

    # Baseline 1: predict the training target mean.
    mean_predictions = np.full(
        len(actual),
        data.y_train.mean(),
    )

    # Baseline 2: linear regression.
    linear_model = LinearRegression()
    linear_model.fit(data.X_train, data.y_train.ravel())
    linear_predictions = linear_model.predict(data.X_test)

    # Calculate evaluation metrics.
    results = {}

    for name, predictions in [
        ("neural_network", neural_predictions),
        ("training_mean", mean_predictions),
        ("linear_regression", linear_predictions),
    ]:
        results[name] = {
            "mae": float(mean_absolute_error(actual, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        }

    # Save test metrics.
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    with (reports_dir / "metrics.json").open("w") as file:
        json.dump(results, file, indent=4)

    print(f"\nBest epoch: {best_epoch}")
    print("\nTest-set comparison:")

    for name, metrics in results.items():
        print(f"{name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    print("\nMetrics saved to reports/metrics.json")
