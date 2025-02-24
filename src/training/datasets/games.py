import os
import h5py
import numpy as np
from typing import Any, Dict
from torch.utils.data import Dataset
from dotenv import load_dotenv
from tqdm import tqdm


class GamesDataset(Dataset):
    def __init__(
        self,
        hdf5_filename: str,
        transform=None,
        min_duration_sec: int = 900,
    ):
        """
        Args:
            hdf5_filename: Path to your HDF5 file.
            transform: Optional transform to be applied on a sample.
            min_duration_sec: Only games with duration >= this (in seconds) will be used to compute normalization parameters.
        """
        self.hdf5_filename = hdf5_filename
        self.transform = transform

        # Open the HDF5 file and load game IDs from the 'frames' group.
        with h5py.File(self.hdf5_filename, "r") as hf:
            if "games" not in hf:
                raise ValueError(
                    f"Group 'games' not found in file '{self.hdf5_filename}'"
                )
            self.game_ids = list(hf["games"].keys())

        # Compute global normalization parameters using all features in the frame vectors.
        print("Computing global normalization parameters...")
        self.norm_means, self.norm_stds = self.compute_global_norm_params(
            min_duration_sec
        )
        print("Global normalization parameters computed.")

    def compute_global_norm_params(self, min_duration_sec: int):
        """
        Computes global normalization parameters (mean and std) for each feature column
        based on the last frame of each game with duration >= min_duration_sec.

        Returns:
            norm_means, norm_stds: 1D NumPy arrays of shape (feature_dim,)
        """
        rep_frames = []
        with h5py.File(self.hdf5_filename, "r") as hf:
            frames_group = hf["frames"]
            for game_id in tqdm(frames_group, desc="Processing games for norm params"):
                ds = frames_group[game_id]
                game_duration = ds.attrs.get("game_duration", 0)
                if game_duration < min_duration_sec:
                    continue
                frames = ds[:]  # shape: (num_frames, feature_dim)
                if frames.shape[0] == 0:
                    continue
                rep_frame = frames[-1, :]  # use the last frame as representative
                rep_frames.append(rep_frame)

        if not rep_frames:
            raise ValueError(
                "No games meet the minimum duration criteria for normalization."
            )

        rep_array = np.vstack(rep_frames)
        feature_dim = rep_array.shape[1]
        norm_means = np.mean(rep_array, axis=0).astype(np.float32)
        norm_stds = np.std(rep_array, axis=0).astype(np.float32)
        # Avoid division by zero:
        norm_stds[norm_stds == 0] = 1.0

        return norm_means, norm_stds

    def __len__(self):
        return len(self.game_ids)

    def __getitem__(self, idx):
        game_id = self.game_ids[idx]
        with h5py.File(self.hdf5_filename, "r") as hf:
            # Load frame data and its attributes from the 'frames' group.
            frames_ds = hf["frames"][game_id]
            frames = frames_ds[:]  # shape: (num_frames, feature_dim)
            game_duration = frames_ds.attrs.get("game_duration", 0)
            early_surrender = bool(frames_ds.attrs.get("early_surrender", 0))
            surrender = bool(frames_ds.attrs.get("surrender", 0))
            blue_win = bool(frames_ds.attrs.get("blue_win", 0))

            # Load champions from the 'champions' group.
            champions = hf["champions"][game_id][:]
            # Load items from the 'items_per_frame' group.
            items = hf["items_per_frame"][game_id][:]

        # Normalize all features in the frame data.
        normalized_frames = frames.copy().astype(np.float32)
        normalized_frames = (normalized_frames - self.norm_means) / self.norm_stds

        sample = {
            "game_id": game_id,
            "frames": normalized_frames,
            "game_duration": game_duration,
            "early_surrender": early_surrender,
            "surrender": surrender,
            "blue_win": blue_win,
            "champions": champions,  # shape: (2, 5)
            "items": items,  # shape: (num_frames, 60)
            "norm_means": self.norm_means,
            "norm_stds": self.norm_stds,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


# Example usage:
if __name__ == "__main__":
    load_dotenv()
    hdf5_filename = "games_data.h5"
    dataset = GamesDataset(hdf5_filename=hdf5_filename)
    print("Total games:", len(dataset))

    sample_game = dataset[0]
    print("Game ID:", sample_game["game_id"])
    print("Frames shape:", sample_game["frames"].shape)
    print("Champions:", sample_game["champions"])
    print("Items shape:", sample_game["items"].shape)
    print("Game duration:", sample_game["game_duration"])
    print("Global norm means (first 10 features):", sample_game["norm_means"][:10])
    print("Global norm stds (first 10 features):", sample_game["norm_stds"][:10])
