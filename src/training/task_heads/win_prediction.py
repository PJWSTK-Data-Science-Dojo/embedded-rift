import torch
import torch.nn as nn

class WinPredictionHead(nn.Module):
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
            nn.Linear(hidden_dim, 1)  # Output is a single scalar logit.
        )

    def forward(self, global_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_repr: Tensor of shape (batch_size, d_model) representing the global summary.
        Returns:
            win_logit: Tensor of shape (batch_size, 1) with the predicted win logit.
        """
        win_logit = self.net(global_repr)
        return win_logit

# Example usage:
if __name__ == "__main__":
    batch_size, d_model = 8, 256
    dummy_global_repr = torch.randn(batch_size, d_model)
    win_head = WinPredictionHead(d_model=d_model)
    win_logit = win_head(dummy_global_repr)
    print("Win logit shape:", win_logit.shape)  # Expected: (8, 1)
