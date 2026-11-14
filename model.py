"""PyTorch dataset, data loader, and neural network."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class BikeDataset(Dataset):
    """Convert prepared NumPy arrays into PyTorch tensors."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 2026,
) -> DataLoader:
    """Create a reproducible DataLoader."""
    dataset = BikeDataset(features, targets)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


class BikeNet(nn.Module):
    """Neural network for predicting hourly bike rentals."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return predicted rental counts."""
        return self.network(features)
