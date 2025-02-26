import math
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from training.datasets.games import GamesDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from training.transforms.multitask_transform import (
    MultitaskTransform,
    OutcomePredictionTransform,
    MaskValuesTransform,
    MaskFramesTransform,
    NextFramePredictionTransform,
)

# fmt: off

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiTaskTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int = 548,  # Continuous features per frame.
        d_model: int = 256,
        champions_model: int = 128,
        items_model: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        use_eos_token: bool = True,
        num_champions: int = 200,  # Total number of unique champion IDs (vocabulary size).
        num_items: int = 256,  # Total number of unique item IDs.
        n_champions_in_game: int = 10,  # Actual number of champions in a game.
        item_slots: int = 60,  # Total number of item IDs per frame (e.g. 10 champions * 6 items each).
    ):
        """
        Args:
            feature_dim: Number of continuous features per frame.
            d_model: Transformer embedding dimension.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            use_cls_token: Whether to use a CLS token for classification.
            num_champions: Vocabulary size for champion IDs.
            num_items: Vocabulary size for item IDs.
            item_slots: Number of item slots per frame.
        """
        super().__init__()
        self.use_cls_token = use_cls_token
        self.use_eos_token = use_eos_token
        self.item_slots = item_slots
        self.items_model = items_model
        self.champions_model = champions_model
        # Project continuous features from the frame into d_model.
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Embedding layers for champions and items.
        self.champion_embedding = nn.Embedding(num_champions, champions_model)
        self.item_embedding = nn.Embedding(num_items, items_model)

        # CLS and EOS tokens.
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        if self.use_eos_token:
            self.eos_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=70)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Projection layers:
        self.champion_proj = nn.Linear(n_champions_in_game * champions_model, d_model)

        num_groups = item_slots // 6  # should be 10 if item_slots is 60.
        self.item_proj = nn.Linear(num_groups * items_model, d_model)

        self.comb_proj = nn.Linear(3 * d_model, d_model)

        # Task-specific heads:
        self.frame_pred_head = nn.Linear(d_model, feature_dim)
        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self, frames: torch.Tensor, champions: torch.Tensor, items: torch.Tensor
    ):
        batch_size, seq_len, _ = frames.size()
        # fmt: off
        # Project continuous features.
        cont_embeds = self.input_proj(frames)  # (batch, seq_len, d_model)

        # Process champion embeddings.
        champ_embeds = self.champion_embedding(champions)  # (batch, n_champions_in_game, champions_model)
        champ_embeds = champ_embeds.view(batch_size, -1)     # (batch, n_champions_in_game * champions_model)
        champ_embeds = champ_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, n_champions_in_game * champions_model)
        champ_embeds = self.champion_proj(champ_embeds)  # (batch, seq_len, d_model)

        # Process item embeddings.
        item_embeds = self.item_embedding(items)  # (batch, seq_len, item_slots, items_model)
        num_groups = items.size(2) // 6  # should be 10 if item_slots is 60.
        item_embeds = item_embeds.view(batch_size, seq_len, num_groups, 6, self.items_model)
        item_embeds = item_embeds.mean(dim=3)  # (batch, seq_len, num_groups, items_model)
        item_embeds = item_embeds.view(batch_size, seq_len, num_groups * self.items_model)  # flatten groups
        item_embeds = self.item_proj(item_embeds)  # (batch, seq_len, d_model)

        # Combine continuous, champion, and item embeddings.
        x_embed = torch.cat([cont_embeds, champ_embeds, item_embeds], dim=-1)  # (batch, seq_len, 3*d_model)
        x_embed = self.comb_proj(x_embed)  # (batch, seq_len, d_model)

        # Prepend CLS token.
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
            x_embed = torch.cat([cls_tokens, x_embed], dim=1)  # (batch, 1+seq_len, d_model)

        # Append EOS token.
        if self.use_eos_token:
            eos_tokens = self.eos_token.expand(batch_size, -1, -1)
            x_embed = torch.cat([x_embed, eos_tokens], dim=1)

        # Add positional encoding.
        x_embed = self.pos_encoder(x_embed)
        x_embed = x_embed.transpose(0, 1)  # (seq_len, batch, d_model)
        x_encoded = self.transformer_encoder(x_embed)
        x_encoded = x_encoded.transpose(0, 1)  # (batch, seq_len(+tokens), d_model)

        # Remove the CLS token for frame-level predictions.
        seq_output = x_encoded[:, 1:, :] if self.use_cls_token else x_encoded

        # Next Frame Prediction:
        if seq_output.size(1) > 2:
            next_frame_input = seq_output[:, : -2, :]
            next_frame_pred = self.frame_pred_head(next_frame_input)
        else:
            next_frame_pred = None

        # Outcome Prediction:
        if self.use_cls_token:
            cls_rep = x_encoded[:, 0, :]
        else:
            cls_rep = x_encoded[:, 0, :]
        outcome_logits = self.classifier(cls_rep).squeeze(-1)

        return next_frame_pred, outcome_logits


#########################################
# Loss Functions
#########################################


def compute_combined_loss(
    frames, next_frame_pred, outcome_logits, sample, lambda_next=1.0, lambda_outcome=1.0
):
    # Next Frame Prediction Loss.
    if next_frame_pred is not None:
        target_next_frames = frames[:, 1:, :]
        loss_next = F.mse_loss(next_frame_pred, target_next_frames)
    else:
        loss_next = 0.0

    # Outcome Prediction Loss.
    target_outcome = sample["outcome"].to(outcome_logits.device)
    loss_outcome = F.binary_cross_entropy_with_logits(outcome_logits, target_outcome)

    total_loss = lambda_next * loss_next + lambda_outcome * loss_outcome
    return total_loss, loss_next, loss_outcome


#########################################
# DataLoader and Collate
#########################################


def custom_collate_fn(batch):
    # Pad frames.
    frames_list = [torch.tensor(item["frames"], dtype=torch.float32) for item in batch]
    padded_frames = pad_sequence(frames_list, batch_first=True)
    # Pad items.
    items_list = [torch.tensor(item["items"], dtype=torch.long) for item in batch]
    padded_items = pad_sequence(items_list, batch_first=True, padding_value=0)
    # Stack champions (assumed fixed size per sample).
    champions_list = [
        torch.tensor(item["champions"], dtype=torch.long) for item in batch
    ]
    champions_tensor = torch.stack(champions_list)
    collated = {
        "frames": padded_frames,
        "items": padded_items,
        "champions": champions_tensor,
    }
    # Optionally include masked frames if available.
    if "frames_masked" in batch[0]:
        masked_frames_list = [
            (
                torch.tensor(item["frames_masked"], dtype=torch.float32)
                if not torch.is_tensor(item["frames_masked"])
                else item["frames_masked"]
            )
            for item in batch
        ]
        padded_masked = pad_sequence(masked_frames_list, batch_first=True)
        collated["frames_masked"] = padded_masked

    # Process outcome and blue_win if available.
    for key in ["outcome", "blue_win"]:
        if key in batch[0]:
            try:
                collated[key] = torch.stack([torch.tensor(item[key]) for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


def get_dataloaders(hdf5_filename, transform, batch_size=4):
    full_dataset = GamesDataset(hdf5_filename=hdf5_filename, transform=transform)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )
    return train_loader, val_loader, test_loader, full_dataset

#########################################
# Training and Evaluation Functions
#########################################
def train_one_epoch(model, dataloader, optimizer, device, lambda_next, lambda_outcome):
    model.train()
    total_loss = 0.0
    total_next_loss = 0.0
    total_outcome_loss = 0.0
    correct_outcomes = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training"):
        frames = batch["frames_masked"].to(device)
        champions = batch["champions"].to(device)
        items = batch["items"].to(device)
        outcome = batch["outcome"].to(device)
        optimizer.zero_grad()
        next_frame_pred, outcome_logits = model(frames, champions, items)
        loss, loss_next, loss_outcome = compute_combined_loss(frames, next_frame_pred, outcome_logits, batch, lambda_next, lambda_outcome)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_next_loss += loss_next.item() if isinstance(loss_next, torch.Tensor) else loss_next
        total_outcome_loss += loss_outcome.item() if isinstance(loss_outcome, torch.Tensor) else loss_outcome

        predicted = (torch.sigmoid(outcome_logits) >= 0.5).float()
        correct_outcomes += (predicted.cpu() == outcome.cpu()).sum().item()
        total_samples += outcome.size(0)
    avg_loss = total_loss / len(dataloader)
    avg_next_loss = total_next_loss / len(dataloader)
    avg_outcome_loss = total_outcome_loss / len(dataloader)
    acc = correct_outcomes / total_samples
    return avg_loss, avg_next_loss, avg_outcome_loss, acc

def evaluate_model(model, dataloader, device, lambda_next, lambda_outcome):
    
    model.eval()
    total_loss = 0.0
    total_next_loss = 0.0
    total_outcome_loss = 0.0
    count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            frames = batch["frames_masked"].to(device)
            champions = batch["champions"].to(device)
            items = batch["items"].to(device)
            outcome = batch["outcome"].to(device)

            next_frame_pred, outcome_logits = model(frames, champions, items)
            loss, loss_next, loss_outcome = compute_combined_loss(frames, next_frame_pred, outcome_logits, batch, lambda_next, lambda_outcome)
            total_loss += loss.item()
            total_next_loss += loss_next.item() if isinstance(loss_next, torch.Tensor) else loss_next
            total_outcome_loss += loss_outcome.item() if isinstance(loss_outcome, torch.Tensor) else loss_outcome
            count += 1

            preds = torch.sigmoid(outcome_logits).cpu().numpy()
            targets = outcome.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    avg_loss = total_loss / count if count > 0 else 0
    avg_next_loss = total_next_loss / count if count > 0 else 0
    avg_outcome_loss = total_outcome_loss / count if count > 0 else 0

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_preds)
    else:
        auc = 0.0

    return avg_loss, avg_next_loss, avg_outcome_loss, auc


#########################################
# Argument Parsing and Main Function
#########################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiTaskTransformer.")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--hdf5-file", type=str, default="games_data.h5", help="Path to HDF5 dataset.")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("-ln", "--lambda-next", type=float, default=1.0, help="Weight for next frame loss.")
    parser.add_argument("-lo", "--lambda-outcome", type=float, default=1.0, help="Weight for outcome loss.")
    parser.add_argument("-log", "--logdir", type=str, default="runs/timeline_transformer", help="Tensorboard log directory.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("-m", "--masking", type=float, default=0.0, help="Probability of masking frames.")
    parser.add_argument("-cls", "--use-cls-token", action="store_true", help="Use CLS token in the model.")
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv()
    
    # Build transform.
    transform = MultitaskTransform(
        next_frame_transform=NextFramePredictionTransform(),
        mask_values_transform=MaskValuesTransform(),
        mask_frames_transform=MaskFramesTransform(mask_frame_prob=args.masking),
        outcome_transform=OutcomePredictionTransform(),
    )
    
    # Get dataloaders.
    train_loader, val_loader, test_loader, full_dataset = get_dataloaders(args.hdf5_file, transform, batch_size=args.batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Extract dimensions.
    sample_game = full_dataset[0]
    feature_dim = sample_game["frames"].shape[-1]
    item_slots = sample_game["items"].shape[-1]
    
    # Build model.
    model = MultiTaskTransformer(
        feature_dim=feature_dim,
        d_model=256,
        champions_model=128,
        items_model=64,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_cls_token=args.use_cls_token,
        use_eos_token=True,
        num_champions=200,
        num_items=256,
        n_champions_in_game=10,
        item_slots=item_slots,
    )
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    writer = SummaryWriter(log_dir=args.logdir)
    
    num_epochs = args.epochs
    best_val_loss = float("inf")
    no_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_next, train_outcome, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, args.lambda_next, args.lambda_outcome
        )
        scheduler.step()
        
        writer.add_scalar("Loss/Train_Next", train_next, epoch)
        writer.add_scalar("Loss/Train_Outcome", train_outcome, epoch)
        
        val_loss, val_next, val_outcome, val_auc = evaluate_model(
            model, val_loader, device, args.lambda_next, args.lambda_outcome
        )
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Val AUC {val_auc:.4f}")
        writer.add_scalar("Loss/Val_Next", val_next, epoch)
        writer.add_scalar("Loss/Val_Outcome", val_outcome, epoch)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("AUC/Val", val_auc, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")
        else:
            no_improvement += 1
        
        if no_improvement >= args.patience:
            print("Early stopping triggered.")
            break
        
        torch.save(model.state_dict(), f"training_checkpoints/checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved for epoch {epoch+1}.")

    test_loss, test_next, test_outcome, test_auc = evaluate_model(
        model, test_loader, device, args.lambda_next, args.lambda_outcome
    )
    print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
