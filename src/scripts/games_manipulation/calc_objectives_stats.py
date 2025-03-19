import numpy as np
from tqdm import tqdm
from utils.hdf5 import HDF5Database

def calc_norms(objectives: np.array) -> tuple:
    norms = np.zeros((objectives.shape[0], objectives.shape[1] * 2))
    stds = np.zeros((objectives.shape[0], objectives.shape[1] * 2))
    
    return norms, stds


if __name__ == '__main__':
    with HDF5Database(db_dir="data/db", compression="gzip") as db:
        # Remove any pre-existing normalization parameters.
        if "norm_params" in db.team_objectives_file:
            del db.player_timeline_file["norm_params"]
        
        max_seq_len = 67
        counts = np.zeros((max_seq_len, 18), dtype=np.int32)
        sums = np.zeros((max_seq_len, 18), dtype=np.float64)
        sumsq = np.zeros((max_seq_len, 18), dtype=np.float64)

        # Iterate over all games and update the accumulators.
        for game_id in tqdm(db.games_group):
            objectives = db.get_team_objectives(game_id)  # shape: (seq_len, 18)
            objectives = objectives[:]
            seq_len = objectives.shape[0]
            # Vectorized update for all frames in this game:
            counts[:seq_len, :] += 1
            sums[:seq_len, :] += objectives
            sumsq[:seq_len, :] += objectives ** 2
            
        # Compute per-frame means and standard deviations in a vectorized manner.
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
        mean_sq = np.divide(sumsq, counts, out=np.zeros_like(sumsq), where=counts != 0)
        variance = mean_sq - means ** 2
        variance = np.clip(variance, a_min=0, a_max=None)
        stds = np.sqrt(variance)

        # Save the computed normalization parameters into the HDF5 file.
        norm_group = db.team_objectives_file.create_group("norm_params")
        norm_group.create_dataset("means", data=means)
        norm_group.create_dataset("std", data=stds)
        
        
        # Compute global normalization parameters if any data was processed.
        norm_group = db.team_objectives_file["norm_params"]
        means = norm_group["means"][:]
        stds = norm_group["std"][:]
        print("Global - Shape of computed means:", means.shape)
        print("Global - Frame 15 means (selected features):", means[15])
        print("Global - Frame 15 stds (selected features):", stds[15])
