import torch
import torch.nn as nn

from training.utils.transformer import PositionalEncoding

class TimelineReconstructionHead(nn.Module):
    def __init__(self, d_model: int, feature_dim: int, max_seq_len: int, hidden_dim: int = 256, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the input global representation.
            feature_dim: Dimension of the original frame features (target output).
            max_seq_len: Maximum sequence length for timeline reconstruction.
            hidden_dim: Dimension of the hidden layer in the decoder.
            dropout: Dropout probability applied in the positional encoding.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        # Learnable query embeddings for each time step.
        self.queries = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Positional encoding applied to the queries.
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_seq_len)
        
        # A simple MLP to decode combined queries into frame features.
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, global_repr: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            global_repr: Tensor of shape (batch_size, d_model) representing the global summary.
            seq_len: The desired sequence length for reconstruction (<= max_seq_len).
        Returns:
            timeline_recon: Tensor of shape (batch_size, seq_len, feature_dim) with the reconstructed timeline.
        """
        batch_size = global_repr.size(0)
        # Get the first seq_len learned queries and expand to batch dimension.
        queries = self.queries[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len, d_model)
        # Apply positional encoding to the queries.
        queries = self.pos_encoder(queries)
        # Expand the global representation to match the sequence length.
        global_expanded = global_repr.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, d_model)
        # Combine the queries with the global representation.
        combined = queries + global_expanded
        # Decode into frame features.
        timeline_recon = self.decoder(combined)  # (batch, seq_len, feature_dim)
        return timeline_recon

# Example usage:
if __name__ == "__main__":
    batch_size = 8
    d_model = 256
    feature_dim = 30
    max_seq_len = 35
    seq_len = 35  # desired reconstruction length

    dummy_global_repr = torch.randn(batch_size, d_model)
    timeline_recon_head = TimelineReconstructionHead(d_model=d_model, feature_dim=feature_dim, max_seq_len=max_seq_len)
    reconstructed_timeline = timeline_recon_head(dummy_global_repr, seq_len)
    print("Reconstructed timeline shape:", reconstructed_timeline.shape)  # Expected: (8, 35, 30)
