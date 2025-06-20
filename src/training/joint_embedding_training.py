from pathlib import Path
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time

from training.datasets.player_timeline import PlayerTimelineDataset
from training.datasets.transforms.pt_transform import PlayerTimelineTransform
from training.model import (
    PlayerTimelineSummaryModel,
    collate_fn
)
from training.models.joint_embedding_training_models import TimelineEncoder, ChampionEncoder, ContextEncoder, Predictor
from datetime import datetime

def info_nce_loss(y, y_hat, temperature=0.1):
    cosine_sim = F.normalize(y) @ F.normalize(y_hat).T
    labels = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
    mask = torch.eye(cosine_sim.shape[0], dtype=bool, device=cosine_sim.device)
    correct_mean = cosine_sim[mask].mean()
    incorrect_mean = cosine_sim[~mask].mean()
    return F.cross_entropy(cosine_sim / temperature, labels), correct_mean, incorrect_mean

#########################################
# Training and Evaluation Loops
#########################################
def train_one_epoch(timeline_encoder, champion_encoder, context_encoder, predictor, dataloader, optimizer, device):
    champion_encoder.train()
    context_encoder.train()
    timeline_encoder.train()
    predictor.train()
    total_loss = 0.0
    total_correct_target_similarity = 0.0
    total_incorrect_target_similarity = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        frames = batch["frames"].to(device)

        champion_target = batch["champion"].to(device).unsqueeze(-1)
        ally_champions = batch["ally_champions"].to(device)
        enemy_champions = batch["enemy_champions"].to(device)
        
        optimizer.zero_grad()

        timeline_emb = timeline_encoder(frames)

        champion_target_expanded = champion_target.expand(-1, ally_champions.size(1))
        ally_mask = ally_champions != champion_target_expanded
        ally_champions_wo_target = ally_champions[ally_mask].reshape(-1, ally_champions.size(1) - 1)
        
        champions = torch.cat([champion_target, ally_champions_wo_target, enemy_champions], dim=1)
        champions_emb = champion_encoder(champions)

        champion_target_emb = champions_emb[:, 0]
        ally_champions_emb = champions_emb[:, 1:5]
        enemy_champions_emb = champions_emb[:, 5:10]

        context_emb = context_encoder(ally_champions_emb, enemy_champions_emb)

        champion_target_emb_pred = predictor(context_emb, timeline_emb)

        loss, correct_mean, incorrect_mean = info_nce_loss(champion_target_emb, champion_target_emb_pred)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct_target_similarity += correct_mean.item()
        total_incorrect_target_similarity += incorrect_mean.item()

        mean_difference = correct_mean - incorrect_mean
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "correct_target": f"{correct_mean.item():.4f}", 
            "incorrect_target": f"{incorrect_mean.item():.4f}",
            "mean_difference": f"{mean_difference.item():.4f}"
        })
    avg_loss = total_loss / len(dataloader)
    avg_correct_target_similarity = total_correct_target_similarity / len(dataloader)
    avg_incorrect_target_similarity = total_incorrect_target_similarity / len(dataloader)
    return avg_loss, avg_correct_target_similarity, avg_incorrect_target_similarity

def evaluate(timeline_encoder, champion_encoder, context_encoder, predictor, dataloader, device):
    champion_encoder.eval()
    context_encoder.eval()
    timeline_encoder.eval()
    predictor.eval()
    total_loss = 0.0
    total_correct_target_similarity = 0.0
    total_incorrect_target_similarity = 0.0
    
    pbar = tqdm(dataloader, desc="Evaluation")
    for batch in pbar:
        frames = batch["frames"].to(device)

        champion_target = batch["champion"].to(device).unsqueeze(-1)
        ally_champions = batch["ally_champions"].to(device)
        enemy_champions = batch["enemy_champions"].to(device)
        
        timeline_emb = timeline_encoder(frames)

        champion_target_expanded = champion_target.expand(-1, ally_champions.size(1))
        ally_mask = ally_champions != champion_target_expanded
        ally_champions_wo_target = ally_champions[ally_mask].reshape(-1, ally_champions.size(1) - 1)
        
        champions = torch.cat([champion_target, ally_champions_wo_target, enemy_champions], dim=1)
        champions_emb = champion_encoder(champions)

        champion_target_emb = champions_emb[:, 0]
        ally_champions_emb = champions_emb[:, 1:5]
        enemy_champions_emb = champions_emb[:, 5:10]

        context_emb = context_encoder(ally_champions_emb, enemy_champions_emb)

        champion_target_emb_pred = predictor(context_emb, timeline_emb)

        loss, correct_mean, incorrect_mean = info_nce_loss(champion_target_emb, champion_target_emb_pred)

        total_loss += loss.item()
        total_correct_target_similarity += correct_mean.item()
        total_incorrect_target_similarity += incorrect_mean.item()

        mean_difference = correct_mean - incorrect_mean
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "correct_target": f"{correct_mean.item():.4f}", 
            "incorrect_target": f"{incorrect_mean.item():.4f}",
            "mean_difference": f"{mean_difference.item():.4f}"
        })
    avg_loss = total_loss / len(dataloader)
    avg_correct_target_similarity = total_correct_target_similarity / len(dataloader)
    avg_incorrect_target_similarity = total_incorrect_target_similarity / len(dataloader)
    return avg_loss, avg_correct_target_similarity, avg_incorrect_target_similarity


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
    parser = argparse.ArgumentParser(description="Train Champion embedding model.")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("-h5", type=str, default="data/db", help="Path to HDF5 db.") 
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("-log", "--logdir", type=str, default="joint_embedding_training", help="Tensorboard log directory.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--model_name", type=str, default="joint_embedding_best_models", help="Name of the directory with best performing models from this experiment")
    return parser.parse_args()

def train(dataset, args):
    num_cores = min(8, os.cpu_count())
    print(f"Number of CPU cores: {num_cores}")
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)
    print(f"Using {num_cores} threads for PyTorch.")
    train_loader, val_loader, test_loader, full_dataset = get_dataloaders(dataset, batch_size=args.batch_size, workers=num_cores // 2)
        
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    print("Using device:", device)

    emb_dim = 512

    sample_game = full_dataset[0]
    feature_dim = sample_game["frames"].shape[-1]

    timeline_encoder = TimelineEncoder(
        feature_dim=feature_dim,
        d_model=emb_dim, 
        num_head=8, 
        num_layers=2, 
        dropout=0.1
    )
    timeline_encoder.to(device)
    
    champion_encoder = ChampionEncoder(
        num_champions=170,
        champion_emb_dim=emb_dim,
        d_model=1024,
        dropout=0.1
    )
    champion_encoder.to(device)

    context_encoder = ContextEncoder(
        d_model=emb_dim,
        num_head=8,
        num_layers=2,
        use_attn_pooling=True
    )
    context_encoder.to(device)

    predictor = Predictor(
        d_model=emb_dim,
        use_FiLM=True  # FiLM (Feature-wise Linear Modulation, https://arxiv.org/pdf/1709.07871) turned off for now 
    )
    predictor.to(device)

    models = {
        'timeline_encoder': timeline_encoder, 
        'champion_encoder': champion_encoder, 
        'context_encoder': context_encoder, 
        'predictor': predictor
    }

    checkpoint_dir = Path(f"checkpoints/{args.logdir}/")  
    writer_dir = Path(f"logs/{args.logdir}/")
    best_model_dir = Path(f"best_models/{args.model_name}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(list(timeline_encoder.parameters()) + 
                                 list(champion_encoder.parameters()) + 
                                 list(context_encoder.parameters()) + 
                                 list(predictor.parameters()), lr=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    writer = SummaryWriter(log_dir=writer_dir)
    
    num_epochs = args.epochs
    best_val_loss = float("inf")
    no_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_correct_similarity_mean, train_incorrect_similarity_mean = train_one_epoch(timeline_encoder, champion_encoder, context_encoder, predictor, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("CorrectSimMean/Train", train_correct_similarity_mean, epoch)
        writer.add_scalar("IncorrectSimMean/Train", train_incorrect_similarity_mean, epoch)

        val_loss, val_correct_similarity_mean, val_incorrect_similarity_mean = evaluate(timeline_encoder, champion_encoder, context_encoder, predictor, val_loader, device)
        print(f"Epoch {epoch+1}: Val Loss {val_loss:.4f}")
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("CorrectSimMean/Val", val_correct_similarity_mean, epoch)
        writer.add_scalar("IncorrectSimMean/Val", val_incorrect_similarity_mean, epoch)

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            for name, model in models.items():
                torch.save(model.state_dict(), best_model_dir / f"joint_embedding_best_model_{name}.pth")
            print("Best model saved.")
        else:
            no_improvement += 1
        
        if no_improvement >= args.patience:
            print("Early stopping triggered.")
            break
        
        for name, model in models.items():
                torch.save(model.state_dict(), checkpoint_dir / f"checkpoint_epoch_{epoch+1}_{name}.pth")
        print(f"Checkpoint saved for epoch {epoch+1}.")
        
    writer.flush()
    writer.close()

def main():
    args = parse_args()
    load_dotenv()
    print(f"Using HDF5 directory: {args.h5}")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    
    # Build our transform.
    transform = PlayerTimelineTransform(mask_frame_prob=0)
    
    # Get dataloaders.
    with PlayerTimelineDataset(db_dir=args.h5, transform=transform) as dataset:
        train(dataset, args)

if __name__ == "__main__":
    main()
