from pathlib import Path
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse

# Import our custom modules.
from training.datasets.timeline import TimelineDataset
from training.transforms.player_timeline import PlayerTimelineTransform
from training.model import (
    MultiTaskTimelineModel,
    collate_fn,  # our custom collate function that batches our keys.
    compute_loss
)

#########################################
# Training and Evaluation Loops
#########################################
def train_one_epoch(model, dataloader, optimizer, device, lambda_next):
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
    }
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move data to device.
        frames = batch["frames"].to(device)
        # Use the target champion from the batch; here we remap the key to "champion"
        champion_target = batch["champion"].to(device)
        composition_ids = batch["composition_champion_ids"].to(device)
        
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
        outputs = model(frames, champion_target, composition_ids)
        loss, loss_details = compute_loss(outputs, targets, lambda_next=lambda_next)
        loss.backward()
        optimizer.step()
        
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

def evaluate_model(model, dataloader, device, lambda_next):
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
    }
    
    count = 0

    pbar = tqdm(dataloader, desc="Evaluation")
    with torch.no_grad():
        for batch in pbar:
            frames = batch["frames"].to(device)
            champion_target = batch["champion"].to(device)
            composition_ids = batch["composition_champion_ids"].to(device)
            
            targets = {
                "duration": batch["duration"].to(device),
                "frames": batch["frames"].to(device),
                "original_frames": batch["original_frames"].to(device),
                "champion": champion_target,
                "position": batch["position"].to(device),
                "side": batch["side"].to(device),
                "win": batch["win"].to(device),
            }
            
            outputs = model(frames, champion_target, composition_ids)
            loss, loss_details = compute_loss(outputs, targets, lambda_next=lambda_next)
            
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

def get_dataloaders(dataset, batch_size=4):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
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
    train_loader, val_loader, test_loader, full_dataset = get_dataloaders(dataset, batch_size=args.batch_size)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Extract feature dimension from one sample.
    sample_game = full_dataset[0]
    feature_dim = sample_game["frames"].shape[-1]
    
    # Build our multi-task model.
    model = MultiTaskTimelineModel(
        feature_dim=feature_dim,
        d_model=256,
        num_champions=200,
        champion_embedding_dim=128,
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
    # Potencjalna zmiana schedulera CosineAnnealingLR Reducelronplateau
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    writer = SummaryWriter(log_dir=writer_dir)
    
    num_epochs = args.epochs
    best_val_loss = float("inf")
    no_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_loss_details = train_one_epoch(model, train_loader, optimizer, device, args.lambda_next)
        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Recon Timeline Loss {train_loss_details.get('loss_timeline', 0.0):.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        for key, value in train_loss_details.items():
            writer.add_scalar(f"Loss/Train_{key}", value, epoch)

        
        val_loss, val_loss_details = evaluate_model(model, val_loader, device, args.lambda_next)
        print(f"Epoch {epoch+1}: Val Loss {val_loss:.4f}, Val Recon Timeline Loss {val_loss_details.get('loss_timeline', 0.0):.4f}")
        writer.add_scalar("Loss/Val", val_loss, epoch)
        for key, value in val_loss_details.items():
            writer.add_scalar(f"Loss/Val_{key}", value, epoch)
        
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

    test_loss, test_loss_details = evaluate_model(model, test_loader, device, args.lambda_next)
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
    with TimelineDataset(db_dir=args.h5, transform=transform) as dataset:
        train(dataset, args)

if __name__ == "__main__":
    main()
