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
from pathlib import Path


def write_champions_to_hdf5(game_input: Dict[str, Any], hf: h5py.File) -> None:
    """
    Writes the champions' data into the HDF5 file.
    """
    collection = "champions"
    game_id = game_input["game_id"]
    blue_champions = game_input["blue_champions"]
    red_champions = game_input["red_champions"]

    champions_group = hf[collection]

    # Create a dataset for this game using its game_id.
    ds = champions_group.create_dataset(
        game_id,
        data=[blue_champions, red_champions],
        dtype=np.int32,
        compression="gzip",
    )
    # print(f"Stored champions for game {game_id}")


def write_game_metadata_to_hdf5(game_input: Dict[str, Any], hf: h5py.File) -> None:
    """
    Writes the game's metadata into the HDF5 file.
    """
    collection = "games"
    game_id = game_input["game_id"]
    game_duration = game_input["game_duration"]
    early_surrender = int(game_input["early_surrender"])
    surrender = int(game_input["surrender"])
    blue_win = int(game_input["blue_win"])
    platform = game_input["platform"]
    season = game_input["season"]
    patch = game_input["patch"]

    metadata_group = hf[collection]

    # Create a dataset for this game using its game_id.
    ds = metadata_group.create_dataset(
        game_id,
        data=[game_duration, early_surrender, surrender, blue_win],
        dtype=np.int32,
        compression="gzip",
    )
    ds.attrs["platform"] = platform
    ds.attrs["season"] = season
    ds.attrs["patch"] = patch
    # print(f"Stored metadata for game {game_id}")


def write_frames_to_hdf5(game_input: Dict[str, Any], hf: h5py.File) -> None:
    """
    Writes the frames' data into the HDF5 file.
    """
    collection = "frames"
    game_id = game_input["game_id"]
    frames = game_input["frames"]
    platform = game_input["platform"]
    season = game_input["season"]
    patch = game_input["patch"]

    first_frame = frames[0]
    keys_order = list(first_frame.keys())

    frame_vectors = []
    for frame in frames:
        vector = [float(frame.get(key, 0)) for key in keys_order]
        frame_vectors.append(vector)
    frame_array = np.array(frame_vectors, dtype=np.float32)

    frames_group = hf[collection]

    # Create a dataset for this game using its game_id.
    ds = frames_group.create_dataset(
        game_id, data=frame_array, compression="gzip", compression_opts=4
    )

    ds.attrs["game_duration"] = game_input["game_duration"]
    ds.attrs["early_surrender"] = int(game_input["early_surrender"])
    ds.attrs["surrender"] = int(game_input["surrender"])
    ds.attrs["blue_win"] = int(game_input["blue_win"])
    ds.attrs["platform"] = platform
    ds.attrs["season"] = season
    ds.attrs["patch"] = patch


def write_items_to_hdf5(game_input: Dict[str, Any], hf: h5py.File) -> None:
    """
    Writes the items' data into the HDF5 file.
    """
    game_id = game_input["game_id"]
    items_per_frame: List[List[int]] = game_input["items_per_frame"]
    collection = "items_per_frame"
    items_group = hf[collection]

    ds = items_group.create_dataset(
        game_id, data=items_per_frame, dtype=np.int32, compression="gzip"
    )


# ---------------------------
# HDF5 writing functions
# ---------------------------
def write_game_to_hdf5(game_input: Dict[str, Any], hf: h5py.File) -> None:
    """
    Writes one game's data (a flattened sequence of frames and metadata) into the HDF5 file.
    The game is stored as its own dataset under the group 'games'.
    """
    game_id = game_input["game_id"]

    if game_id in hf["games"]:
        return

    write_game_metadata_to_hdf5(game_input, hf)
    write_frames_to_hdf5(game_input, hf)
    write_champions_to_hdf5(game_input, hf)
    write_items_to_hdf5(game_input, hf)


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
    path = Path(hdf5_filename)
    files_exist = path.exists()

    # Open HDF5 file in append mode so that we don't overwrite existing data.
    with h5py.File(hdf5_filename, "a") as hf:
        if not files_exist:
            hf.create_group("games")
            hf.create_group("frames")
            hf.create_group("champions")
            hf.create_group("items_per_frame")

        cursor = collection.find({})  # Optionally add filters to your query.
        for i, game in tqdm(enumerate(cursor)):
            game_input = extract_game_data(game)
            write_game_to_hdf5(game_input, hf)
            if i > 10_000:
                break

    print(f"Finished processing games into {hdf5_filename}")


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    print(mongo_uri)
    db_name = "embedded-rift"  # Update as needed
    collection_name = "games"  # Update as needed
    hdf5_filename = "data/games_data.h5"  # Output HDF5 file

    process_games_to_hdf5(mongo_uri, db_name, collection_name, hdf5_filename)
