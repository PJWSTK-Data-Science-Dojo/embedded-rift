import os
import h5py
import numpy as np
from typing import Any, Dict, List
from pymongo import MongoClient
from dotenv import load_dotenv
from utils.training.game_to_frames import extract_game_data
from tqdm import tqdm
import concurrent.futures
import itertools


def batched(iterable, n):
    """
    Batch data into lists of length n. The last batch may be shorter.
    Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


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

    # Use the first frame to determine the vector order (assumes all frames have the same keys).
    first_frame = frames[0]
    keys_order = list(first_frame.keys())

    # Convert each frame (dict) into a vector following keys_order.
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


def process_games_to_hdf5_parallel(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    hdf5_filename: str,
    num_workers: int = 4,
    batch_size: int = 1000,
) -> None:
    """
    Connects to MongoDB, processes games in batches to avoid loading all data into RAM,
    extracts game data in parallel using multiple processes, and writes each game to an HDF5 file.
    Checks for duplicates to avoid re-writing existing games.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Open the HDF5 file in append mode.
    with h5py.File(hdf5_filename, "a") as hf:
        # Get or create the "games" group.
        if "games" in hf:
            games_group = hf["games"]
        else:
            games_group = hf.create_group("games")

        cursor = collection.find({})  # Optionally add filters to your query.
        total_processed = 0
        # Process documents in batches.
        for batch in batched(cursor, batch_size):
            print(f"Processing a batch of {len(batch)} games...")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                # Process each game in the current batch in parallel.
                results = list(
                    tqdm(executor.map(extract_game_data, batch), total=len(batch))
                )
            # Write each extracted game sequentially (to avoid HDF5 concurrency issues).
            for game_input in results:
                game_id = game_input["game_id"]
                if game_id in games_group:
                    # Skip if the game is already stored.
                    continue
                write_game_to_hdf5(game_input, hf)
            total_processed += len(batch)
            print(f"Total games processed so far: {total_processed}")
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

    process_games_to_hdf5_parallel(mongo_uri, db_name, collection_name, hdf5_filename)
