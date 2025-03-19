import torch
import torch.nn as nn

class DurationPredictionHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 128):
        """
        Args:
            d_model: Dimension of the input global representation.
            hidden_dim: Dimension of the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output is a single scalar: the game duration.
        )

    def forward(self, timeline_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_repr: Tensor of shape (batch_size, d_model) representing the global summary.
        Returns:
            duration_pred: Tensor of shape (batch_size, 1) with the predicted game duration.
        """
        duration_pred = self.net(timeline_cls)
        return duration_pred