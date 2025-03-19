import h5py
import numpy as np
from tqdm import tqdm

from utils.hdf5 import HDF5Database

def save_norm_params_for_players(db: HDF5Database, max_seq_length=67):
    """
    For each group under "players", compute the per-frame mean and standard deviation for each feature,
    ignoring NaN values. Then, for features whose index is NOT divisible by 3, set the mean to 0 and std to 1,
    so that these features do not normalize the data.
    
    Each game data is of shape (seq_len, no_features), where seq_len can vary.
    For each frame index (0 to max_seq_length-1), the function computes the mean and std dev
    over all games that have that frame index.
    
    In addition, a global normalization group is computed that aggregates the statistics 
    over all groups combined.
    
    The computed means and standard deviations for each group and the global group are 
    saved under the "norm_params" group in the h5 file.
    """
    players_file = db.player_timeline_file
    if "players" not in players_file:
        raise ValueError("No 'players' collection found in the database.")
    
    players_group = players_file["players"]
    
    # Remove any existing "norm_params" group so that we start fresh.
    if "norm_params" in players_file:
        del players_file["norm_params"]
    norm_params_grp = players_file.create_group("norm_params")
    
    # Global accumulators for all groups combined.
    global_sum = None
    global_sum_sq = None
    global_count = None

    # Process each of the 10 groups (group_0 to group_9)
    for group_idx in range(10):
        subgroup_name = f"group_{group_idx}"
        if subgroup_name not in players_group:
            continue  # Skip if this subgroup doesn't exist.
        subgroup = players_group[subgroup_name]
        
        # Initialize accumulators for this subgroup.
        sum_arr = None
        sum_sq_arr = None
        count_arr = None
        
        games = list(subgroup.keys())
        for game_player_id in tqdm(games, desc=f"Processing {subgroup_name}"):
            data = subgroup[game_player_id][:]
            seq_len, no_features = data.shape
            effective_len = min(seq_len, max_seq_length)
            
            # Initialize subgroup accumulators if not already initialized.
            if sum_arr is None:
                sum_arr = np.zeros((max_seq_length, no_features), dtype=np.float64)
                sum_sq_arr = np.zeros((max_seq_length, no_features), dtype=np.float64)
                count_arr = np.zeros((max_seq_length, no_features), dtype=np.int64)
            
            # Use a mask to ignore NaN values.
            data_effective = data[:effective_len]
            mask = ~np.isnan(data_effective)  # shape: (effective_len, no_features)
            sum_arr[:effective_len] += np.where(mask, data_effective, 0)
            sum_sq_arr[:effective_len] += np.where(mask, data_effective**2, 0)
            count_arr[:effective_len] += mask.astype(np.int64)
            
            # Also accumulate global values.
            if global_sum is None:
                global_sum = np.zeros((max_seq_length, no_features), dtype=np.float64)
                global_sum_sq = np.zeros((max_seq_length, no_features), dtype=np.float64)
                global_count = np.zeros((max_seq_length, no_features), dtype=np.int64)
            global_sum[:effective_len] += np.where(mask, data_effective, 0)
            global_sum_sq[:effective_len] += np.where(mask, data_effective**2, 0)
            global_count[:effective_len] += mask.astype(np.int64)
        
        # Compute subgroup means and stds, avoiding division by zero.
        mean_arr = np.divide(sum_arr, count_arr, out=np.zeros_like(sum_arr), where=count_arr != 0)
        mean_sq = np.divide(sum_sq_arr, count_arr, out=np.zeros_like(sum_sq_arr), where=count_arr != 0)
        variance = mean_sq - mean_arr**2
        variance = np.clip(variance, a_min=0, a_max=None)
        std_arr = np.sqrt(variance)
        std_arr[count_arr == 0] = 1.0  # Default std for frames with no data.
        
        # Set means and stds for features whose indices are NOT divisible by 3:
        no_features = mean_arr.shape[1]
        indices = np.arange(no_features)
        mask_non_div3 = (indices % 3) != 0
        mean_arr[:, mask_non_div3] = 0.0
        std_arr[:, mask_non_div3] = 1.0
        
        # Save subgroup normalization parameters.
        group_norm = norm_params_grp.create_group(subgroup_name)
        group_norm.create_dataset("means", data=mean_arr, compression="gzip")
        group_norm.create_dataset("std", data=std_arr, compression="gzip")
    
    # Compute global normalization parameters if any data was processed.
    if global_sum is not None:
        global_mean = np.divide(global_sum, global_count, out=np.zeros_like(global_sum), where=global_count != 0)
        global_mean_sq = np.divide(global_sum_sq, global_count, out=np.zeros_like(global_sum_sq), where=global_count != 0)
        global_variance = global_mean_sq - global_mean**2
        global_variance = np.clip(global_variance, a_min=0, a_max=None)
        global_std = np.sqrt(global_variance)
        global_std[global_count == 0] = 1.0
        
        # Apply the same mask for features not divisible by 3.
        no_features = global_mean.shape[1]
        indices = np.arange(no_features)
        mask_non_div3 = (indices % 3) != 0
        global_mean[:, mask_non_div3] = 0.0
        global_std[:, mask_non_div3] = 1.0
        
        global_group = norm_params_grp.create_group("global")
        global_group.create_dataset("means", data=global_mean, compression="gzip")
        global_group.create_dataset("std", data=global_std, compression="gzip")
    
    print("Normalization parameters (mean and std) saved successfully.")
    return


# Example usage:
if __name__ == '__main__':
    with HDF5Database(db_dir="data/db", compression="gzip") as db:
        # Remove any pre-existing normalization parameters.
        if "norm_params" in db.player_timeline_file:
            del db.player_timeline_file["norm_params"]
        
        # Compute and save the normalization parameters.
        save_norm_params_for_players(db, max_seq_length=67)
        
        # For example, load and inspect normalization parameters for group_0 and the global stats:
        norm_group = db.player_timeline_file["norm_params"]["group_0"]
        means = norm_group["means"][:]
        stds = norm_group["std"][:]
        print("Group_0 - Shape of computed means:", means.shape)
        print("Group_0 - Frame 20 means (selected features):", means[20])
        print("Group_0 - Frame 20 stds (selected features):", stds[20])
        
        global_group = db.player_timeline_file["norm_params"]["global"]
        global_means = global_group["means"][:]
        global_stds = global_group["std"][:]
        print("Global - Shape of computed means:", global_means.shape)
        print("Global - Frame 15 means (selected features):", global_means[15])
        print("Global - Frame 15 stds (selected features):", global_stds[15])
