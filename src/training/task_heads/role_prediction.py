import torch
import torch.nn as nn

class RolePredictionHead(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 5, hidden_dim: int = 128):
        """
        Args:
            d_model: Dimension of the input global representation.
            num_positions: Total number of unique positions (e.g., 5 for top, jungle, mid, ADC, support).
            hidden_dim: Dimension of the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_positions)
        )

    def forward(self, global_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_repr: Tensor of shape (batch_size, d_model) representing the global summary.
        Returns:
            position_logits: Tensor of shape (batch_size, num_positions) with raw class scores.
        """
        position_logits = self.net(global_repr)
        return position_logits

# Example usage:
if __name__ == "__main__":
    batch_size, d_model, num_positions = 8, 256, 5
    dummy_global_repr = torch.randn(batch_size, d_model)
    position_head = RolePredictionHead(d_model=d_model, num_positions=num_positions)
    position_logits = position_head(dummy_global_repr)
    print("Position logits shape:", position_logits.shape)  # Expected: (8, 5)
