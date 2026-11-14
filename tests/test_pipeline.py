"""Tests for the bike rental prediction pipeline."""

import numpy as np
import pytest
import torch
from torch import nn

from bikes_data import make_rentals
from model import BikeDataset, BikeNet, make_loader
from prepare import (
    NUMERIC_FEATURES,
    prepare_data,
    split_data,
)
from train import evaluate, train_model


def test_split_sizes_and_no_overlap():
    """Check 60/20/20 split and ensure rows do not overlap."""
    frame = make_rentals()
    train, val, test = split_data(frame)

    assert len(train) == 3600
    assert len(val) == 1200
    assert len(test) == 1200

    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)


def test_scaler_fits_training_rows_only():
    """Check numeric scaler means come from training rows only."""
    frame = make_rentals()
    train, _, _ = split_data(frame)
    data = prepare_data(frame)

    scaler = data.preprocessor.named_transformers_["numeric"]
    expected_means = train[NUMERIC_FEATURES].mean().to_numpy()

    np.testing.assert_allclose(scaler.mean_, expected_means)


@pytest.mark.parametrize("sample_count", [5, 10, 20])
def test_dataset_length(sample_count):
    """Check Dataset length for different sample counts."""
    features = np.ones((sample_count, 4), dtype=np.float32)
    targets = np.ones((sample_count, 1), dtype=np.float32)

    dataset = BikeDataset(features, targets)

    assert len(dataset) == sample_count


def test_dataset_tensor_shapes_and_dtypes():
    """Check Dataset returns correctly shaped float32 tensors."""
    features = np.ones((10, 4), dtype=np.float32)
    targets = np.ones((10, 1), dtype=np.float32)

    dataset = BikeDataset(features, targets)
    x, y = dataset[0]

    assert x.shape == (4,)
    assert y.shape == (1,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_shuffled_loader_is_reproducible():
    """Check shuffled loaders produce the same order with the same seed."""
    features = np.zeros((20, 2), dtype=np.float32)
    targets = np.arange(20, dtype=np.float32).reshape(-1, 1)

    loader_one = make_loader(features, targets, batch_size=5, shuffle=True, seed=42)
    loader_two = make_loader(features, targets, batch_size=5, shuffle=True, seed=42)

    order_one = torch.cat([y for _, y in loader_one]).flatten()
    order_two = torch.cat([y for _, y in loader_two]).flatten()

    assert torch.equal(order_one, order_two)
    assert not torch.equal(order_one, torch.arange(20, dtype=torch.float32))


def test_model_output_dimensions():
    """Check the network returns one prediction per row."""
    model = BikeNet(input_size=8, hidden_size=16)
    inputs = torch.randn(5, 8)

    predictions = model(inputs)

    assert predictions.shape == (5, 1)


def test_optimizer_changes_weights():
    """Check an optimizer step changes model parameters."""
    model = BikeNet(input_size=4, hidden_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    features = torch.randn(10, 4)
    targets = torch.randn(10, 1)

    before = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad()
    loss = loss_function(model(features), targets)
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters())
    )


def test_model_can_memorize_small_dataset():
    """Check the network can learn a small synthetic dataset."""
    torch.manual_seed(42)

    features = torch.randn(16, 2)
    targets = 2 * features[:, :1] - features[:, 1:2]

    model = BikeNet(input_size=2, hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    with torch.no_grad():
        initial_loss = loss_function(model(features), targets).item()

    for _ in range(300):
        optimizer.zero_grad()
        loss = loss_function(model(features), targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = loss_function(model(features), targets).item()

    assert final_loss < initial_loss * 0.1


def test_evaluation_does_not_change_weights():
    """Check evaluation leaves model weights unchanged."""
    data = prepare_data()
    loader = make_loader(data.X_val, data.y_val)
    model = BikeNet(input_size=data.X_train.shape[1])
    loss_function = nn.MSELoss()

    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    evaluate(model, loader, loss_function)

    after = model.state_dict()

    for key, value in before.items():
        assert torch.equal(value, after[key])


def test_best_epoch_weights_are_restored():
    """Check returned model matches the best validation loss."""
    model, history, best_epoch = train_model(epochs=5)

    data = prepare_data()
    loader = make_loader(data.X_val, data.y_val)
    loss_function = nn.MSELoss()

    restored_loss, _ = evaluate(model, loader, loss_function)
    expected_best_loss = min(item["val_loss"] for item in history)

    assert best_epoch == min(history, key=lambda item: item["val_loss"])["epoch"]
    assert restored_loss == pytest.approx(expected_best_loss, rel=1e-5)
