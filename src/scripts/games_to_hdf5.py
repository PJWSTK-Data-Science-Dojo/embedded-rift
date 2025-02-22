import os
import json
import h5py
import numpy as np
from typing import Any, Dict, List
from pymongo import MongoClient
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from utils.training.game_to_frames import extract_game_data
from tqdm import tqdm


# ---------------------------
# HDF5 writing functions
# ---------------------------
def write_game_to_hdf5(
    game_input: Dict[str, Any], hf: h5py.File, games_group_name: str = "games"
) -> None:
    """
    Writes one game's data (a flattened sequence of frames and metadata) into the HDF5 file.
    The game is stored as its own dataset under the group 'games'.
    """
    game_id = game_input["game_id"]
    frames = game_input["frames"]
    if not frames:
        print(f"Game {game_id} has no frames, skipping.")
        return

    # Use the first frame to determine the vector order.
    # (We assume each flattened frame has the same keys.)
    first_frame = frames[0]
    keys_order = list(first_frame.keys())  # relying on insertion order

    # Convert each frame (dict) into a vector (list) following keys_order.
    frame_vectors = []
    for frame in frames:
        vector = [float(frame.get(key, 0)) for key in keys_order]
        frame_vectors.append(vector)
    frame_array = np.array(frame_vectors, dtype=np.float32)

    # Ensure the "games" group exists.
    if games_group_name in hf:
        games_group = hf[games_group_name]
    else:
        games_group = hf.create_group(games_group_name)

    # Create a dataset for this game using its game_id.
    ds = games_group.create_dataset(
        game_id, data=frame_array, compression="gzip", compression_opts=4
    )
    ds.attrs["game_duration"] = game_input["game_duration"]
    ds.attrs["early_surrender"] = int(game_input["early_surrender"])
    ds.attrs["surrender"] = int(game_input["surrender"])
    ds.attrs["blue_win"] = int(game_input["blue_win"])
    # print(f"Stored game {game_id} with shape {frame_array.shape}")


def process_games_to_hdf5(
    mongo_uri: str, db_name: str, collection_name: str, hdf5_filename: str
) -> None:
    """
    Connects to MongoDB, iterates over games, extracts game data, and writes each game
    to an HDF5 file. Checks if the game is already in the file and skips it if so.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Open HDF5 file in append mode so that we don't overwrite existing data.
    with h5py.File(hdf5_filename, "a") as hf:
        cursor = collection.find({})  # Optionally add filters to your query.
        for game in tqdm(cursor):
            game_input = extract_game_data(game)
            write_game_to_hdf5(game_input, hf)
    print(f"Finished processing games into {hdf5_filename}")


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    db_name = "embedded-rift"  # Update as needed
    collection_name = "games"  # Update as needed
    hdf5_filename = "games_data.h5"  # Output HDF5 file

    process_games_to_hdf5(mongo_uri, db_name, collection_name, hdf5_filename)
