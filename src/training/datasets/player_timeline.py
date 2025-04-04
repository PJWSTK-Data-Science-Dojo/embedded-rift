import numpy as np
from typing import Any, Dict, Self, Optional
from torch.utils.data import Dataset
from dotenv import load_dotenv
from utils.champion import CHAMP_ID_TO_INDEX
from utils.hdf5 import HDF5Database
from training.datasets.transforms.pt_transform import PlayerTimelineTransform

INDEX_TO_POS = {
    0: "TOP",
    1: "JUNGLE",
    2: "MIDDLE",
    3: "BOTTOM",
    4: "UTILITY",
}

POS_TO_INDEX = {v: k for k, v in INDEX_TO_POS.items()}

class PlayerTimelineDataset(Dataset):
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
        
        
    def normalize(self, frames: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """
        Normalize the frames using the provided means and stds.
        
        Args:
            frames: Array of frames to normalize.
            means: Mean values for normalization.
            stds: Standard deviation values for normalization.
        
        Returns:
            Normalized frames.
        """
        seq_len, _ = frames.shape
        epsilon = 1e-8
        return (frames - means[:seq_len]) / (stds[:seq_len] + epsilon)
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        game_id, player_idx = self.entries[idx]

        # Retrieve player timeline.
        player_timeline_ds = self.db.get_player_timeline(game_id, player_idx)
        player_timeline = player_timeline_ds[:]
        
        # Retrieve game attributes.
        game_duration = player_timeline_ds.attrs.get("game_duration", 0)
        early_surrender = bool(player_timeline_ds.attrs.get("early_surrender", 0))
        surrender = bool(player_timeline_ds.attrs.get("surrender", 0))
        blue_win = bool(player_timeline_ds.attrs.get("blue_win", 0))
        
        # Retrieve team objectives.
        objectives_ds = self.db.get_team_objectives(game_id)
        team_objectives = objectives_ds[:]  # shape: (num_frames, objective_dim)
                
        norm_group = self.db.team_objectives_file["norm_params"]        
        team_objectives = self.normalize(team_objectives, norm_group["means"][:], norm_group["std"][:])
        
        # Retrieve champions for the game.
        champions_ds = self.db.get_champions(game_id)
        champions_arr = champions_ds[:]  # shape: (2, 5)
        champions = np.array(
            [CHAMP_ID_TO_INDEX[champion] for team in champions_arr for champion in team],
            dtype=np.int32,
        )
        

        team = "blue" if player_idx < 5 else "red"
        side = 0 if team == "blue" else 1

        
        player_champion = player_timeline_ds.attrs["player_champion"]
        champion = CHAMP_ID_TO_INDEX[player_champion]
        
        win = True if (player_idx < 5 and blue_win) or (player_idx >= 5 and not blue_win) else False
        
        # Retrieve normalization statistics.
        means, std = self.db.get_timeline_norms()
        normalized_frames = self.normalize(player_timeline, means, std)
        
        position = POS_TO_INDEX[player_timeline_ds.attrs["position"]]
        # Compute side: 0 for blue, 1 for red.


        # Create the sample dictionary with keys expected by the training pipeline.
        sample = {
            "game_id": game_id,
            "player_idx": player_idx,
            "position": position,           # integer index for position
            "champions": champions,          # shape: (10,)
            "frames": normalized_frames,          # shape: (seq_len, feature_dim)
            "team_objectives": team_objectives,      # optional, for additional supervision or analysis
            "duration": round(2 * game_duration / 60) / 2,              # scalar value (can be normalized later if needed)
            "champion": champion,     # integer index for champion prediction
            "win": int(win),                        # binary (0 or 1)
            "side": side,                           # binary: 0 (blue) or 1 (red)
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample




# Example usage:
if __name__ == "__main__":
    load_dotenv()
    
    with PlayerTimelineDataset(db_dir="data/db", transform=PlayerTimelineTransform(0.1)) as dataset:
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