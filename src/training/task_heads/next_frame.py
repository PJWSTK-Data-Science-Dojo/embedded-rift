import torch
import torch.nn as nn

class NextFramePredictionHead(nn.Module):
    def __init__(self, d_model: int, feature_dim: int, hidden_dim: int = 256):
        """
        Args:
            d_model: Dimension of the input frame tokens from the transformer.
            feature_dim: Dimension of the original frame features (target output).
            hidden_dim: Dimension of the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_tokens: Tensor of shape (batch, seq_len, d_model)
                          representing the frame tokens from the transformer.
        Returns:
            next_frame_pred: Tensor of shape (batch, seq_len, feature_dim)
                             containing the predicted next frame features for each token.
        """
        next_frame_pred = self.net(frame_tokens)
        return next_frame_pred

# Example usage:
if __name__ == "__main__":
    batch_size, seq_len, d_model, feature_dim = 8, 35, 256, 30
    dummy_frame_tokens = torch.randn(batch_size, seq_len, d_model)
    next_frame_head = NextFramePredictionHead(d_model=d_model, feature_dim=feature_dim)
    predicted_next_frames = next_frame_head(dummy_frame_tokens)
    print("Predicted next frame shape:", predicted_next_frames.shape)  # Expected: (8, 35, 30)
