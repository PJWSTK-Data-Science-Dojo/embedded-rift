import math
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from training.datasets.games import GamesDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.nn.functional as F

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()


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
        # For champions: flatten the per-game champion embeddings (n_champions_in_game * champions_model) to d_model.
        self.champion_proj = nn.Linear(n_champions_in_game * champions_model, d_model)
        # For items: group items by 6 -> num_groups = item_slots // 6, then flatten to (num_groups * items_model) -> d_model.
        num_groups = item_slots // 6  # should be 10 if item_slots is 60.
        self.item_proj = nn.Linear(num_groups * items_model, d_model)
        # After concatenating continuous, champion, and item representations (3*d_model) project back to d_model.
        self.comb_proj = nn.Linear(3 * d_model, d_model)

        # Task-specific heads:
        self.frame_pred_head = nn.Linear(d_model, feature_dim)
        self.masked_pred_head = nn.Linear(d_model, feature_dim)
        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self, frames: torch.Tensor, champions: torch.Tensor, items: torch.Tensor
    ):
        """
        Args:
            frames: Tensor of shape (batch, seq_len, feature_dim) for continuous features.
            champions:
            items: Tensor of shape (batch, seq_len, item_slots)
        Returns:
            next_frame_pred: Prediction for next frame features (autoregressive).
            masked_value_pred: Prediction for masked value reconstruction.
            outcome_logits: Logits for game outcome classification.
        """
        # fmt: off
        batch_size, seq_len, _ = frames.size()

        # Project continuous features.
        cont_embeds = self.input_proj(frames)  # (batch, seq_len, d_model)

        # Process champion embeddings:
        # champions: (batch, n_champions_in_game)
        champ_embeds = self.champion_embedding(champions)  # (batch, n_champions_in_game, champions_model)
        champ_embeds = champ_embeds.view(batch_size, -1)     # (batch, n_champions_in_game * champions_model)
        # Broadcast to each frame:
        champ_embeds = champ_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, n_champions_in_game * champions_model)
        champ_embeds = self.champion_proj(champ_embeds)  # (batch, seq_len, d_model)

        # Process item embeddings:
        # items: (batch, seq_len, item_slots)  where item_slots is 60
        item_embeds = self.item_embedding(items)  # (batch, seq_len, item_slots, items_model)
        # Group items by 6:
        num_groups = items.size(2) // 6  # should be 10
        item_embeds = item_embeds.view(batch_size, seq_len, num_groups, 6, self.items_model)
        item_embeds = item_embeds.mean(dim=3)  # (batch, seq_len, num_groups, items_model)
        item_embeds = item_embeds.view(batch_size, seq_len, num_groups * self.items_model)  # flatten groups
        item_embeds = self.item_proj(item_embeds)  # (batch, seq_len, d_model)

        # Combine continuous, champion, and item embeddings:
        x_embed = torch.cat([cont_embeds, champ_embeds, item_embeds], dim=-1)  # (batch, seq_len, 3*d_model)
        x_embed = self.comb_proj(x_embed)  # (batch, seq_len, d_model)

        # Optionally, prepend a CLS token.
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(
                batch_size, -1, -1
            )  # (batch, 1, d_model)
            x_embed = torch.cat(
                [cls_tokens, x_embed], dim=1
            )  # (batch, 1+seq_len, d_model)

        if self.use_eos_token:
            eos_tokens = self.eos_token.expand(batch_size, -1, -1)
            x_embed = torch.cat([x_embed, eos_tokens], dim=1)

        # Add positional encodings.
        x_embed = self.pos_encoder(x_embed)

        # Transformer expects input as (seq_len, batch, d_model).
        x_embed = x_embed.transpose(0, 1)
        x_encoded = self.transformer_encoder(x_embed)
        x_encoded = x_encoded.transpose(0, 1)  # (batch, seq_len (+1), d_model)

        # Remove the CLS token for frame-level predictions.
        seq_output = x_encoded[:, 1:, :] if self.use_cls_token else x_encoded

        # 1. Next Frame Prediction:
        # Ensure there are enough tokens: we need at least 3 tokens (CLS, at least one frame, EOS)
        if seq_output.size(1) > 2:
            # Use the first (T - 2) tokens, where T = seq_output.size(1), so that the prediction length is T - 2.
            # This will match the target shape of frames[:, 1:, :], which is (batch, L - 1, feature_dim) if original L frames.
            next_frame_input = seq_output[:, : -2, :]
            next_frame_pred = self.frame_pred_head(next_frame_input)
        else:
            next_frame_pred = None

        # 2. Masked Value Prediction:
        masked_value_pred = self.masked_pred_head(seq_output)

        # 3. Outcome Prediction:
        if self.use_cls_token:
            cls_rep = x_encoded[:, 0, :]  # (batch, d_model)
        else:
            cls_rep = x_encoded[:, -1, :]  # (batch, d_model)
        outcome_logits = self.classifier(cls_rep).squeeze(-1)

        # fmt: on
        return next_frame_pred, masked_value_pred, outcome_logits


def compute_combined_loss(
    frames, next_frame_pred, masked_value_pred, outcome_logits, sample
):
    # 1. Next Frame Prediction Loss.
    if next_frame_pred is not None:
        target_next_frames = frames[:, 1:, :]
        loss_next = mse_loss(next_frame_pred, target_next_frames)
    else:
        loss_next = 0.0

    # 2. Masked Value Prediction Loss.
    mask = sample["mask_values"]  # Expected shape: (batch, seq_len, feature_dim)
    if mask.sum() > 0:
        pred_masked = masked_value_pred[mask]
        target_masked = frames[mask]
        loss_masked = mse_loss(pred_masked, target_masked)
    else:
        loss_masked = 0.0

    # 3. Outcome Prediction Loss.
    target_outcome = sample["outcome"]  # Expected shape: (batch,)
    loss_outcome = bce_loss(outcome_logits, target_outcome)

    # Combine losses with weights.
    lambda_next = 1.0
    lambda_masked = 1.0
    lambda_outcome = 1.0
    total_loss = (
        lambda_next * loss_next
        + lambda_masked * loss_masked
        + lambda_outcome * loss_outcome
    )

    return total_loss, loss_next, loss_masked, loss_outcome


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_next_loss = 0.0
    total_masked_loss = 0.0
    total_outcome_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            frames = torch.tensor(batch["frames"], dtype=torch.float32).to(device)
            champions = torch.tensor(batch["champions"], dtype=torch.long).to(device)
            items = torch.tensor(batch["items"], dtype=torch.long).to(device)
            # Ensure "mask_values" and "outcome" keys exist.
            if "mask_values" not in batch:
                batch["mask_values"] = torch.zeros_like(frames, dtype=torch.bool).to(
                    device
                )
            if "outcome" not in batch:
                outcome_list = [1.0 if flag else 0.0 for flag in batch["blue_win"]]
                batch["outcome"] = torch.tensor(outcome_list, dtype=torch.float32).to(
                    device
                )
            next_frame_pred, masked_value_pred, outcome_logits = model(
                frames, champions, items
            )
            loss, loss_next, loss_masked, loss_outcome = compute_combined_loss(
                frames, next_frame_pred, masked_value_pred, outcome_logits, batch
            )
            total_loss += loss.item()
            total_next_loss += (
                loss_next.item() if isinstance(loss_next, torch.Tensor) else loss_next
            )
            total_masked_loss += (
                loss_masked.item()
                if isinstance(loss_masked, torch.Tensor)
                else loss_masked
            )
            total_outcome_loss += (
                loss_outcome.item()
                if isinstance(loss_outcome, torch.Tensor)
                else loss_outcome
            )
            count += 1
    avg_loss = total_loss / count
    avg_next = total_next_loss / count
    avg_masked = total_masked_loss / count
    avg_outcome = total_outcome_loss / count
    return avg_loss, avg_next, avg_masked, avg_outcome


def custom_collate_fn(batch):
    """
    Expects each element in batch to be a dict with keys: "frames", "champions", "items", etc.
    Pads sequences (e.g., frames and items) to the maximum length in the batch.
    """
    # Process frames (assumed shape: (seq_len, feature_dim))
    frames_list = [torch.tensor(item["frames"], dtype=torch.float32) for item in batch]
    padded_frames = pad_sequence(
        frames_list, batch_first=True
    )  # (batch, max_seq_len, feature_dim)

    # Process items (assumed shape: (seq_len, item_slots))
    items_list = [torch.tensor(item["items"], dtype=torch.long) for item in batch]
    padded_items = pad_sequence(
        items_list, batch_first=True, padding_value=0
    )  # (batch, max_seq_len, item_slots)

    # For champions: if they are constant per game (shape: (n_champions,)),
    # we can simply stack them if they are the same size.
    champions_list = [
        torch.tensor(item["champions"], dtype=torch.long) for item in batch
    ]
    champions_tensor = torch.stack(champions_list)  # (batch, n_champions)

    # Process additional keys if needed.
    collated = {
        "frames": padded_frames,
        "items": padded_items,
        "champions": champions_tensor,
    }

    # Optionally handle other keys (e.g., "mask_values", "blue_win", "outcome").
    for key in batch[0]:
        if key not in collated:
            try:
                collated[key] = torch.stack([torch.tensor(item[key]) for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]

    return collated


def main():
    load_dotenv()
    hdf5_filename = "games_data.h5"
    full_dataset = GamesDataset(hdf5_filename=hdf5_filename)

    # Split dataset: 80% train, 10% validation, 10% test.
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn
    )

    # Device handling.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Get dimensions from one sample.
    sample_game = full_dataset[0]
    feature_dim = sample_game["frames"].shape[-1]
    champion_const = True  # adjust based on your data.
    item_slots = sample_game["items"].shape[-1]

    # Instantiate the model.
    model = MultiTaskTransformer(
        feature_dim=feature_dim,
        d_model=256,
        champions_model=128,
        items_model=64,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_cls_token=True,
        use_eos_token=True,
        num_champions=200,
        num_items=256,
        n_champions_in_game=10,
        item_slots=item_slots,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Learning rate scheduler: step down every 3 epochs with gamma factor.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct_outcomes = 0
        total_outcomes = 0

        # Use tqdm to monitor training progress.
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            frames = batch["frames"].to(device)
            champions = batch["champions"].to(device)
            items = batch["items"].to(device)
            # Ensure keys "mask_values" and "outcome" exist.
            if "mask_values" not in batch:
                batch["mask_values"] = torch.zeros_like(frames, dtype=torch.bool).to(
                    device
                )
            if "outcome" not in batch:
                outcome_list = [1.0 if flag else 0.0 for flag in batch["blue_win"]]
                batch["outcome"] = torch.tensor(outcome_list, dtype=torch.float32).to(
                    device
                )

            optimizer.zero_grad()
            next_frame_pred, masked_value_pred, outcome_logits = model(
                frames, champions, items
            )
            total_loss_batch, loss_next, loss_masked, loss_outcome = (
                compute_combined_loss(
                    frames, next_frame_pred, masked_value_pred, outcome_logits, batch
                )
            )
            total_loss_batch.backward()
            optimizer.step()

            epoch_loss += total_loss_batch.item()

            # Outcome Accuracy: apply sigmoid and threshold at 0.5.
            predicted = (torch.sigmoid(outcome_logits) >= 0.5).float()
            correct_outcomes += (predicted.cpu() == batch["outcome"].cpu()).sum().item()
            total_outcomes += batch["outcome"].size(0)

            pbar.set_postfix(loss=total_loss_batch.item())

        scheduler.step()  # Update learning rate.

        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_outcomes / total_outcomes

        # Evaluate on validation set.
        val_loss, val_next, val_masked, val_outcome = evaluate(
            model, val_loader, device
        )
        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Outcome Acc = {train_accuracy:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )

        # Save checkpoint.
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved for epoch {epoch+1}.")

    # After training, evaluate on test set.
    test_loss, test_next, test_masked, test_outcome = evaluate(
        model, test_loader, device
    )
    print(
        f"Test Loss: {test_loss:.4f} (Next: {test_next:.4f}, Masked: {test_masked:.4f}, Outcome: {test_outcome:.4f})"
    )
    print("Training completed.")


if __name__ == "__main__":
    main()
