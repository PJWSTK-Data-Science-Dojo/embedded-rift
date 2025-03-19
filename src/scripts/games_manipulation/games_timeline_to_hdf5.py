import json
import orjson
import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

from utils.hdf5 import HDF5Database
from utils.training.frames import extract_frames
from utils.training.player_frames import extract_all_players_frames

def process_games(db_dir: str) -> None:
    """
    Connects to MongoDB, extracts game data using extract_frames and extract_all_players_frames,
    and writes each game's data into the HDF5 database.
    """
    with HDF5Database(db_dir=db_dir, compression="gzip") as hdf_db:
        
        with open("games.jsonl", "r") as f:
            
            for line in tqdm(f):
                game = orjson.loads(line)
                # try:
                # Extract the game input using your custom extractor.
                game_input = extract_frames(game)
                game_id = game_input["game_id"]
                hdf_db.write_items(game_id, game_input)
                # if game_id in hdf_db:
                #     continue
                # # Write game metadata, frames, champions, and objectives.
                # hdf_db.write_game(game_id, game_input)
                # hdf_db.write_frame(game_id, game_input)
                # hdf_db.write_champions(game_id, game_input)
                # hdf_db.write_team_objectives(game_id, game_input)

                # # Extract and write player timelines.
                # players_data = extract_all_players_frames(game_input)
                # for player_idx in players_data:
                #     hdf_db.write_player_timeline(game_id, player_idx, players_data, game_input)

                # except Exception as e:
                #     print(e)
                #     logging.error(f"Error processing game {game.get('game_id', 'unknown')}: {e}")

if __name__ == "__main__":
    load_dotenv()
    db_dir = "data/db"               # Directory for your HDF5 database

    process_games(db_dir)
