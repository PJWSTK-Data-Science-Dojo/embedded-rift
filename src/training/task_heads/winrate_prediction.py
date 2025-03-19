import torch
import torch.nn as nn

class PerFrameWinRatePredictionHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 128):
        """
        Args:
            d_model: Dimension of the input per-frame token.
            hidden_dim: Dimension of the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output is a single scalar win rate per frame.
        )

    def forward(self, timeline_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timeline_tokens: Tensor of shape (batch_size, seq_len, d_model) representing the timeline tokens.
        Returns:
            win_rate_pred: Tensor of shape (batch_size, seq_len, 1) containing the predicted win rate for each frame.
        """
        win_rate_pred = self.net(timeline_tokens)
        return win_rate_pred

# Example usage:
if __name__ == "__main__":
    batch_size, seq_len, d_model = 8, 35, 256
    dummy_timeline_tokens = torch.randn(batch_size, seq_len, d_model)
    per_frame_win_rate_head = PerFrameWinRatePredictionHead(d_model=d_model)
    predicted_win_rate = per_frame_win_rate_head(dummy_timeline_tokens)
    print("Predicted win rate shape:", predicted_win_rate.shape)  # Expected: (8, 35, 1)
