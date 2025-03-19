import numpy as np
from typing import Any, Dict, Self, Optional
from torch.utils.data import Dataset
from dotenv import load_dotenv
from utils.champion import CHAMP_ID_TO_INDEX
from utils.hdf5 import HDF5Database
from training.transforms.player_timeline import PlayerTimelineTransform
class TimelineDataset(Dataset):
    def __init__(
        self,
        db_dir: str = "data/db",
        transform: Optional[Any] = None
    ):
        """
        Args:
            db_dir: Directory where the HDF5 files are stored.
            norm_means: Precomputed normalization means (e.g., shape: [max_seq_len, feature_dim]).
            norm_stds: Precomputed normalization stds (same shape as norm_means).
            transform: Optional transform to be applied on a sample.
            min_duration_sec: Only games with duration >= this (in seconds) will be used.
        """
        self.transform = transform
        self.db = HDF5Database(db_dir=db_dir)
        
        # Open the database and build an index of available player timeline entries.
        # The players data are stored in the "players" group inside the player_timeline_file.
    
    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        return False
        
    def open(self) -> None:
        self.db.open()
        print("Building player timeline index...")
        self.entries = self.db.get_game_player_entries()
    
    def close(self) -> None:
        print("Closing database...")
        self.db.close()
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        game_id, player_idx = self.entries[idx]

        # Retrieve player timeline.
        player_timeline_ds = self.db.get_player_timeline(game_id, player_idx)
        player_timeline = player_timeline_ds[:]
        seq_len = player_timeline.shape[0]        
        
        # Retrieve game attributes.
        game_duration = player_timeline_ds.attrs.get("game_duration", 0)
        early_surrender = bool(player_timeline_ds.attrs.get("early_surrender", 0))
        surrender = bool(player_timeline_ds.attrs.get("surrender", 0))
        blue_win = bool(player_timeline_ds.attrs.get("blue_win", 0))
        
        # Retrieve team objectives.
        objectives_ds = self.db.get_team_objectives(game_id)
        team_objectives = objectives_ds[:]  # shape: (num_frames, objective_dim)
        
        # Normalize team objectives using precomputed normalization parameters if available.
        epsilon = 1e-8
        norm_group = self.db.team_objectives_file["norm_params"]
        team_means = norm_group["means"][:]  # shape: (max_seq_len, objective_dim)
        team_stds = norm_group["std"][:]     # shape: (max_seq_len, objective_dim)
        # Use only the first seq_len rows for the current game.
        team_objectives = (team_objectives - team_means[:seq_len]) / (team_stds[:seq_len] + epsilon)
        
        # Retrieve champions for the game.
        champions_ds = self.db.get_champions(game_id)
        champions_arr = champions_ds[:]  # shape: (2, 5)
        # Flatten into a (10,) array and convert using your champion mapping.
        champions = np.array(
            [CHAMP_ID_TO_INDEX[champion] for team in champions_arr for champion in team],
            dtype=np.int32,
        )
        # Reshape to (2, team_size) – assuming team size is 5.
        composition_champion_ids = champions.reshape(2, 5)
        
        # Retrieve the target champion (the one played by the current player).
        player_champion = player_timeline_ds.attrs["player_champion"]
        target_champion = CHAMP_ID_TO_INDEX[player_champion]
        
        # Determine team: first 5 players are blue, rest are red.
        team = "blue" if player_idx < 5 else "red"
        # Determine win: if blue wins then blue players win, red players lose.
        win = True if (player_idx < 5 and blue_win) or (player_idx >= 5 and not blue_win) else False
        
        # Retrieve normalization statistics.
        means, std = self.db.get_timeline_norms()
        epsilon = 1e-8
        normalized_frames = (player_timeline - means[:seq_len]) / (std[:seq_len] + epsilon)
        
        # Set default position (if you don't have it, you might set a default or derive from other data).
        position = player_idx % 5  # e.g., 0 might represent a default role.
        # Compute side: 0 for blue, 1 for red.
        side = 0 if team == "blue" else 1

        # Create the sample dictionary with keys expected by the training pipeline.
        sample = {
            "game_id": game_id,
            "player_idx": player_idx,
            "frames": normalized_frames,          # shape: (seq_len, feature_dim)
            "duration": game_duration / 60,              # scalar value (can be normalized later if needed)
            "champion": target_champion,     # integer index for champion prediction
            "composition_champion_ids": composition_champion_ids,  # shape: (2, team_size)
            "win": int(win),                        # binary (0 or 1)
            "position": position,           # integer index for position
            "side": side,                           # binary: 0 (blue) or 1 (red)
            "team_objectives": team_objectives,      # optional, for additional supervision or analysis
            "obj_means": team_means,                # normalization statistics (optional)
            "obj_stds": team_stds,                  # normalization statistics (optional)
            "norm_means": means,                    # normalization statistics (optional)
            "norm_stds": std,                       # normalization statistics (optional)
        }
        
        if self.transform:
            sample = self.transform(sample)
        return sample




# Example usage:
if __name__ == "__main__":
    load_dotenv()
    
    with TimelineDataset(db_dir="data/db", transform=PlayerTimelineTransform(0.1)) as dataset:
        print("Total player timeline entries:", len(dataset))
        
        sample = dataset[0]
        print("Frame stats:", sample["frames"].min(), sample["frames"].max(), sample["frames"].mean())
        print("Game ID:", sample["game_id"])
        print("Player Index:", sample["player_idx"])
        print("Frames:", sample["frames"][:10])
        print("Champions:", sample["composition_champion_ids"])
        print("Game duration:", sample["duration"])
        print("Global norm means (first 10 features):", sample["norm_means"][:10])
        print("Global norm stds (first 10 features):", sample["norm_stds"][:10])
        print("Objectives shape:", sample["team_objectives"].shape)