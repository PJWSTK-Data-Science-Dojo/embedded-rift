from pathlib import Path
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
# Import our custom modules.
from training.datasets.player_timeline import PlayerTimelineDataset
from training.datasets.transforms.pt_transform import PlayerTimelineTransform
from training.model import (
    PlayerTimelineSummaryModel,
    collate_fn,  # our custom collate function that batches our keys.
    compute_loss
)

#########################################
# Training and Evaluation Loops
#########################################
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_loss_detials = {
        "loss_duration": 0,
        "loss_next": 0,
        "loss_timeline": 0,
        "loss_champion": 0,
        "loss_position": 0,
        "loss_side": 0,
        "loss_win": 0,
        "loss_win_rate": 0,
        "loss_cham_champion": 0,
        "loss_cham_position": 0,
    }
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move data to device.
        frames = batch["frames"].to(device)
        # Use the target champion from the batch; here we remap the key to "champion"
        champion_target = batch["champion"].to(device)
        ally_champions = batch["ally_champions"].to(device)
        enemy_champions = batch["enemy_champions"].to(device)
        
        # Build targets dictionary. For reconstruction, we use the unmasked original_frames.
        targets = {
            "duration": batch["duration"].to(device),
            "frames": batch["frames"].to(device),
            "original_frames": batch["original_frames"].to(device),
            "champion": champion_target,   # remapped key: champion target.
            "position": batch["position"].to(device),
            "side": batch["side"].to(device),
            "win": batch["win"].to(device),
            # Optionally, if you want per-frame win rate loss using win label, it will be handled in loss function.
        }
        
        optimizer.zero_grad()
        outputs = model(frames, champion_target, ally_champions, enemy_champions)
        # print(outputs)
        # outputs = {key: torch.nan_to_num(value) for key, value in outputs.items()}
        # print(outputs)
        loss, loss_details = compute_loss(outputs, targets)
        # print(loss_details["loss_next"])
        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            print(targets)
            raise Exception()
        
        total_loss += loss.item()
        for k, v in loss_details.items():
            total_loss_detials[k] += v

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "next_loss": f"{loss_details.get('loss_next', 0.0):.4f}",
        })
    avg_loss = total_loss / len(dataloader)
    avg_loss_keys = {k: v / len(dataloader) for k, v in total_loss_detials.items()}
    return avg_loss, avg_loss_keys

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_loss_detials = {
        "loss_duration": 0,
        "loss_next": 0,
        "loss_timeline": 0,
        "loss_champion": 0,
        "loss_position": 0,
        "loss_side": 0,
        "loss_win": 0,
        "loss_win_rate": 0,
        "loss_cham_champion": 0,
        "loss_cham_position": 0,
    }
    
    count = 0

    pbar = tqdm(dataloader, desc="Evaluation")
    with torch.no_grad():
        for batch in pbar:
            frames = batch["frames"].to(device)
            champion_target = batch["champion"].to(device)
            ally_champions = batch["ally_champions"].to(device)
            enemy_champions = batch["enemy_champions"].to(device)
            
            targets = {
                "duration": batch["duration"].to(device),
                "frames": batch["frames"].to(device),
                "original_frames": batch["original_frames"].to(device),
                "champion": champion_target,
                "position": batch["position"].to(device),
                "side": batch["side"].to(device),
                "win": batch["win"].to(device),
            }
            
            outputs = model(frames, champion_target, ally_champions, enemy_champions)
            loss, loss_details = compute_loss(outputs, targets)
            
            total_loss += loss.item()
            for k, v in loss_details.items():
                total_loss_detials[k] += v
            count += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "next_loss": f"{loss_details.get('loss_next', 0.0):.4f}",
            })
    avg_loss = total_loss / count if count > 0 else 0
    avg_loss_keys = {k: v / count for k, v in total_loss_detials.items()}
    return avg_loss, avg_loss_keys 

def get_dataloaders(dataset, batch_size=4, workers=6):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers)
    return train_loader, val_loader, test_loader, dataset

#########################################
# Argument Parsing and Main Function
#########################################
def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Task Timeline Transformer.")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("-h5", type=str, default="data/db", help="Path to HDF5 db.")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("-ln", "--lambda-next", type=float, default=1.0, help="Weight for next frame loss.")
    parser.add_argument("-log", "--logdir", type=str, default="timeline_transformer", help="Tensorboard log directory.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("-m", "--masking", type=float, default=0.0, help="Probability of masking frames.")
    return parser.parse_args()

def train(dataset, args):
    num_cores = min(8, os.cpu_count())
    print(f"Number of CPU cores: {num_cores}")
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)
    print(f"Using {num_cores} threads for PyTorch.")
    train_loader, val_loader, test_loader, full_dataset = get_dataloaders(dataset, batch_size=args.batch_size, workers=num_cores // 2)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Extract feature dimension from one sample.
    sample_game = full_dataset[0]
    feature_dim = sample_game["frames"].shape[-1]
    
    # Build our multi-task model.
    model = PlayerTimelineSummaryModel(
        feature_dim=feature_dim,
        d_model=256,
        num_champions=200,
        champion_embedding_dim=16,
        max_seq_len=70,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        num_positions=5
    )
    model.to(device)
    
    checkpoint_dir = Path(f"checkpoints/{args.logdir}/")
    writer_dir = Path(f"logs/{args.logdir}/")
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Potential change of scheduler CosineAnnealingLR Reducelronplateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",       # We want to reduce LR when validation loss stops decreasing.
        factor=0.5,       # Multiply LR by 0.5 on plateau.
        patience=2,       # Number of epochs with no improvement before reducing LR.
    )
    writer = SummaryWriter(log_dir=writer_dir)
    
    num_epochs = args.epochs
    best_val_loss = float("inf")
    no_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_loss_details = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Recon Timeline Loss {train_loss_details.get('loss_timeline', 0.0):.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        for key, value in train_loss_details.items():
            writer.add_scalar(f"Loss/Train_{key}", value, epoch)

        
        val_loss, val_loss_details = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}: Val Loss {val_loss:.4f}, Val Recon Timeline Loss {val_loss_details.get('loss_timeline', 0.0):.4f}")
        writer.add_scalar("Loss/Val", val_loss, epoch)
        for key, value in val_loss_details.items():
            writer.add_scalar(f"Loss/Val_{key}", value, epoch)
        
        scheduler.step(val_loss)
        # print(f"Learning rate: {scheduler.get_last_lr():.6f}")
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
        
        torch.save(model.state_dict(), checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved for epoch {epoch+1}.")

    test_loss, test_loss_details = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Recon Timeline Loss {test_loss_details.get('loss_timeline', 0.0):.4f}")
    writer.add_scalar("Loss/Test", test_loss, epoch)
    for key, value in test_loss_details.items():
        writer.add_scalar(f"Loss/Test_{key}", value, epoch)
        
    writer.flush()
    writer.close()

def main():
    args = parse_args()
    load_dotenv()
    print(f"Using HDF5 directory: {args.h5}")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    
    # Build our transform.
    transform = PlayerTimelineTransform(mask_frame_prob=args.masking)
    
    # Get dataloaders.
    with PlayerTimelineDataset(db_dir=args.h5, transform=transform) as dataset:
        train(dataset, args)

if __name__ == "__main__":
    main()
