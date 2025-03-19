import torch
import torch.nn as nn

class SidePredictionHead(nn.Module):
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
            nn.Linear(hidden_dim, 2)  # Two classes: blue and red.
        )

    def forward(self, timeline_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timeline_cls: Tensor of shape (batch_size, d_model) representing the global summary.
        Returns:
            side_logits: Tensor of shape (batch_size, 2) with raw class scores for blue/red.
        """
        side_logits = self.net(timeline_cls)
        return side_logits

# Example usage:
if __name__ == "__main__":
    batch_size, d_model = 8, 256
    dummy_global_repr = torch.randn(batch_size, d_model)
    side_head = SidePredictionHead(d_model=d_model)
    side_logits = side_head(dummy_global_repr)
    print("Side logits shape:", side_logits.shape)  # Expected: (8, 2)
