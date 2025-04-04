import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from training.task_heads.champion_prediction import ChampionPredictionHead
from training.task_heads.game_duration import DurationPredictionHead
from training.task_heads.next_frame import NextFramePredictionHead
from training.task_heads.role_prediction import RolePredictionHead
from training.task_heads.side_prediction import SidePredictionHead
from training.task_heads.timeline_reconstruction import TimelineReconstructionHead
from training.task_heads.win_prediction import WinPredictionHead
from training.task_heads.winrate_prediction import PerFrameWinRatePredictionHead
from training.utils.transformer import PositionalEncoding

class PlayerTimelineSummaryModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        num_champions: int,
        champion_embedding_dim: int,
        max_seq_len: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_positions: int = 5
    ):
        super().__init__()
        # Shared transformer backbone (e.g., your Timeline Transformer)
        self.frame_proj = nn.Linear(feature_dim, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        
        # Champion and team tokens processing.
        self.champion_embedding = nn.Embedding(num_champions, champion_embedding_dim)
        self.champion_proj = nn.Linear(champion_embedding_dim, d_model)
        self.team_proj = nn.Linear(5 * champion_embedding_dim, d_model)
        
        self.cls_token = nn.Parameter(torch.zeros(1, d_model))
        
        # Positional encoding.
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer Encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # TaskHeads.
        self.duration_head = DurationPredictionHead(d_model=d_model)
        self.next_frame_head = NextFramePredictionHead(d_model=d_model, feature_dim=feature_dim)
        self.timeline_recon_head = TimelineReconstructionHead(d_model=d_model, feature_dim=feature_dim, max_seq_len=max_seq_len)
        self.champion_head = ChampionPredictionHead(d_model=d_model, num_champions=num_champions)
        self.position_head = RolePredictionHead(d_model=d_model, num_positions=num_positions)
        self.side_head = SidePredictionHead(d_model=d_model)
        self.win_head = WinPredictionHead(d_model=d_model)
        self.per_frame_win_rate_head = PerFrameWinRatePredictionHead(d_model=d_model)

    def forward(self, frames: torch.Tensor, champion_id: torch.Tensor, ally_champions: torch.Tensor, enemy_champions: torch.Tensor):
        """
        Args:
            frames: Tensor of shape (batch, seq_len, feature_dim)
            target_champion_id: Tensor of shape (batch,)
            composition_champion_ids: Tensor of shape (batch, 2, team_size)
        Returns:
            A dictionary with predictions for each task.
        """
        batch_size, seq_len, _ = frames.shape
        recon_seq_len = seq_len
        # Process frame tokens.
        frame_tokens = self.frame_proj(frames)  # (batch, seq_len, d_model)
        
        # Process target champion token.
        target_embed = self.champion_embedding(champion_id)  # (batch, champion_embedding_dim)
        champion_token = self.champion_proj(target_embed)             # (batch, d_model)
        
        # Process team composition token.
        ally_embeds = self.champion_embedding(ally_champions)
        enemy_embeds  = self.champion_embedding(enemy_champions)
        ally_vector = ally_embeds.view(batch_size, -1)  # (batch, team_size * champion_embedding_dim)
        enemy_vector = enemy_embeds.view(batch_size, -1)  # (batch, team_size * champion_embedding_dim)
        ally_token = self.team_proj(ally_vector)
        enemy_token = self.team_proj(enemy_vector)

        
        cls_token = self.cls_token.expand(batch_size, -1).unsqueeze(1)  
        champion_token = champion_token.unsqueeze(1)  # (batch, 1, d_model)
        ally_token = ally_token.unsqueeze(1)      # (batch, 1, d_model)
        enemy_token = enemy_token.unsqueeze(1)    # (batch, 1, d_model)
        tokens = torch.cat([cls_token, champion_token, ally_token, enemy_token, frame_tokens], dim=1)  # (batch, 4+seq_len, d_model)
        
        # Add positional encoding and normalization.
        tokens = self.pos_encoder(tokens)
        tokens = self.token_norm(tokens)
        
        # Pass through transformer encoder.
        encoded = self.transformer_encoder(tokens)  # (batch, 3+seq_len, d_model)
        
        # Extract timeline tokens (skip the first two tokens).
        cls_token, champ_embed, ally_tok, enemy_tok, timeline_tokens = (
            torch.split(encoded, [1, 1, 1, 1, seq_len], dim=1)
        )
        sq_cls = cls_token.squeeze(1)  # (batch, d_model)
        # Get predictions from each head.
        duration_pred = self.duration_head(sq_cls)                       # (batch, 1)
        next_frame_pred = self.next_frame_head(timeline_tokens[:, :-1, :])       # (batch, seq_len-1, feature_dim)
        timeline_recon = self.timeline_recon_head(sq_cls, recon_seq_len)    # (batch, recon_seq_len, feature_dim)
        champion_logits = self.champion_head(sq_cls)                      # (batch, num_champions)
        position_logits = self.position_head(sq_cls)                      # (batch, num_positions)
        side_logits = self.side_head(sq_cls)                              # (batch, 2)
        win_logit = self.win_head(sq_cls)                                  # (batch, 1)
        per_frame_win_rate = self.per_frame_win_rate_head(timeline_tokens)       # (batch, seq_len, 1)
        
        return {
            "duration_pred": duration_pred,
            "next_frame_pred": next_frame_pred,
            "timeline_recon": timeline_recon,
            "champion_logits": champion_logits,
            "position_logits": position_logits,
            "side_logits": side_logits,
            "win_logit": win_logit,
            "per_frame_win_rate": per_frame_win_rate,
            "global_repr": sq_cls,       # Global game summary embedding.
            "champion_embedding": champ_embed,   # Champion representation.
        }
        
        
def compute_loss(outputs: dict, targets: dict,
                           lambda_duration: float = 1.0,
                           lambda_next: float = 1.0,
                           lambda_timeline: float = 1.0,
                           lambda_champion: float = 1.0,
                           lambda_position: float = 1.0,
                           lambda_side: float = 1.0,
                           lambda_win: float = 1.0,
                           lambda_win_rate: float = 1.0):
    """
    Computes a composite loss for multi-task learning.

    Args:
        outputs: dict containing model predictions with keys:
            - "duration_pred": Tensor of shape (batch, 1)
            - "next_frame_pred": Tensor of shape (batch, seq_len-1, feature_dim)
            - "timeline_recon": Tensor of shape (batch, recon_seq_len, feature_dim)
            - "champion_logits": Tensor of shape (batch, num_champions)
            - "position_logits": Tensor of shape (batch, num_positions)
            - "side_logits": Tensor of shape (batch, 2)
            - "win_logit": Tensor of shape (batch, 1)
            - "per_frame_win_rate": Tensor of shape (batch, seq_len, 1)
        targets: dict containing ground truth values with keys:
            - "duration": Tensor of shape (batch,) or (batch, 1)
            - "frames": Tensor of shape (batch, seq_len, feature_dim) used for next frame and timeline reconstruction
            - "champion": Tensor of shape (batch,) containing class indices
            - "position": Tensor of shape (batch,) containing class indices
            - "side": Tensor of shape (batch,) containing class indices (0 or 1)
            - "win": Tensor of shape (batch,) with binary values (0 or 1)
            - "win_rate": Tensor of shape (batch, seq_len) representing target win rates per frame
        lambda_*: Loss weighting factors for each task.

    Returns:
        total_loss: Combined weighted loss.
        loss_details: A dictionary with individual loss values for monitoring.
    """
    # Duration Prediction Loss: Mean Squared Error
    loss_duration = F.mse_loss(outputs["duration_pred"].squeeze(), targets["duration"].float())
    
    # Next Frame Prediction Loss: MSE Loss between predicted next frames and target frames shifted by one time step.
    # Assuming target frames shape is (batch, seq_len, feature_dim) and next_frame_pred predicts for seq_len-1 steps.
    loss_next = F.mse_loss(outputs["next_frame_pred"], targets["original_frames"][:, 1:, :])

    # Whole Timeline Reconstruction Loss: MSE Loss between reconstructed timeline and original frames.
    loss_timeline = F.mse_loss(outputs["timeline_recon"], targets["original_frames"])

    # Champion Prediction Loss: Cross-Entropy Loss.
    loss_champion = F.cross_entropy(outputs["champion_logits"], targets["champion"])
    
    # Position Prediction Loss: Cross-Entropy Loss.
    loss_position = F.cross_entropy(outputs["position_logits"], targets["position"])

    # Side Prediction Loss: Cross-Entropy Loss.
    loss_side = F.cross_entropy(outputs["side_logits"], targets["side"])

    # Win Prediction Loss: Binary Cross-Entropy with Logits.
    loss_win = F.binary_cross_entropy_with_logits(outputs["win_logit"].squeeze(), targets["win"].float())

    # Per-Frame Win Rate Prediction Loss: MSE Loss.
    # Get batch size and sequence length from model output.
    batch_size, seq_len, _ = outputs["per_frame_win_rate"].shape
    # Expand win label (shape: (batch,)) to (batch, seq_len)
    win_target = targets["win"].float().unsqueeze(1).expand(batch_size, seq_len)
    # Compute MSE loss between predicted win rates (squeezed to shape (batch, seq_len)) and the constant win target.
    loss_win_rate = F.mse_loss(outputs["per_frame_win_rate"].squeeze(-1), win_target)

    total_loss = (lambda_duration * loss_duration +
                  lambda_next * loss_next +
                  lambda_timeline * loss_timeline +
                  lambda_champion * loss_champion +
                  lambda_position * loss_position +
                  lambda_side * loss_side +
                  lambda_win * loss_win +
                  lambda_win_rate * loss_win_rate)

    loss_details = {
        "loss_duration": loss_duration.item() if isinstance(loss_duration, torch.Tensor) else loss_duration,
        "loss_next": loss_next.item() if isinstance(loss_next, torch.Tensor) else loss_next,
        "loss_timeline": loss_timeline.item() if isinstance(loss_timeline, torch.Tensor) else loss_timeline,
        "loss_champion": loss_champion.item() if isinstance(loss_champion, torch.Tensor) else loss_champion,
        "loss_position": loss_position.item() if isinstance(loss_position, torch.Tensor) else loss_position,
        "loss_side": loss_side.item() if isinstance(loss_side, torch.Tensor) else loss_side,
        "loss_win": loss_win.item() if isinstance(loss_win, torch.Tensor) else loss_win,
        "loss_win_rate": loss_win_rate.item() if isinstance(loss_win_rate, torch.Tensor) else loss_win_rate,
    }

    return total_loss, loss_details

# ------------------------------
# Custom Collate Function
# ------------------------------
def collate_fn(batch):
    frames_list = [torch.tensor(sample["frames"], dtype=torch.float32) for sample in batch]
    
    orig_frames_list = [torch.tensor(sample["original_frames"], dtype=torch.float32) for sample in batch]
    
    collated = {
        "frames": pad_sequence(frames_list, batch_first=True),
        "original_frames": pad_sequence(orig_frames_list, batch_first=True),
        "champion": torch.tensor([sample["champion"] for sample in batch], dtype=torch.long),
        "ally_champions": torch.stack([torch.tensor(sample["ally_champions"], dtype=torch.long) for sample in batch]),
        "enemy_champions": torch.stack([torch.tensor(sample["enemy_champions"], dtype=torch.long) for sample in batch]),
        "duration": torch.tensor([sample["duration"] for sample in batch], dtype=torch.float32),
        "win": torch.tensor([sample["win"] for sample in batch], dtype=torch.long),
        "position": torch.tensor([sample["position"] for sample in batch], dtype=torch.long),
        "side": torch.tensor([sample["side"] for sample in batch], dtype=torch.long),
    }
    
    return collated

# Example usage:
if __name__ == "__main__":
    batch_size, seq_len, feature_dim = 8, 35, 30
    d_model = 256
    num_champions = 200
    champion_embedding_dim = 128
    max_seq_len = 35
    num_positions = 5

    # Create dummy inputs.
    dummy_frames = torch.randn(batch_size, seq_len, feature_dim)
    dummy_target_champion = torch.randint(0, num_champions, (batch_size,))
    dummy_composition_ids = torch.randint(0, num_champions, (batch_size, 2, 5))  # Assuming team size of 5.
    
    model = PlayerTimelineSummaryModel(
        feature_dim=feature_dim,
        d_model=d_model,
        num_champions=num_champions,
        champion_embedding_dim=champion_embedding_dim,
        max_seq_len=max_seq_len,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        num_positions=num_positions
    )
    
    outputs = model(dummy_frames, dummy_target_champion, dummy_composition_ids)
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
