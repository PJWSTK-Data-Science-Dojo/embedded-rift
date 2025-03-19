import torch
import torch.nn as nn

class ChampionPredictionHead(nn.Module):
    def __init__(self, d_model: int, num_champions: int, hidden_dim: int = 128):
        """
        Args:
            d_model: Dimension of the input global representation.
            num_champions: Total number of unique champions (number of classes).
            hidden_dim: Dimension of the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_champions)
        )

    def forward(self, game_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_repr: Tensor of shape (batch_size, d_model) representing the global summary.
        Returns:
            champion_logits: Tensor of shape (batch_size, num_champions) with raw class scores.
        """
        champion_logits = self.net(game_cls)
        return champion_logits

# Example usage:
if __name__ == "__main__":
    batch_size, d_model, num_champions = 8, 256, 200
    dummy_global_repr = torch.randn(batch_size, d_model)
    champion_head = ChampionPredictionHead(d_model=d_model, num_champions=num_champions)
    champion_logits = champion_head(dummy_global_repr)
    print("Champion logits shape:", champion_logits.shape)  # Expected: (8, 200)
