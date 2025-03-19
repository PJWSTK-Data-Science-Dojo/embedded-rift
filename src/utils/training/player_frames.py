import json
import os
from typing import Any, Dict, List, Iterable

from dotenv import load_dotenv
import numpy as np
from pymongo import MongoClient
from utils.training.frames import extract_frames
from itertools import islice


def split_list(a: List[Any], n: int) -> Iterable[List[Any]]:
    """
    Split list 'a' into 'n' nearly equal parts.
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def extract_all_players_frames(game_input: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extracts and enriches each player's frame data with team and overall percentages.
    
    Returns:
        A dictionary with keys 0-9 representing each player and values being lists of enriched frame data.
    """
    player_frames: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(10)}
    frames: List[Dict[str, Any]] = game_input["frames"]

    for frame in frames:
        # Convert the frame items to a list once
        items = list(frame.items())
        mid = len(items) // 2
        blue_items = items[:mid]
        red_items = items[mid:]
        
        # Split each team's items into 5 players
        blue_players = list(split_list(blue_items, 5))
        red_players = list(split_list(red_items, 5))
        
        # Precompute dictionaries and their corresponding NumPy arrays for blue team
        blue_data = []
        for player in blue_players:
            player_dict = dict(player)
            arr = np.array(list(player_dict.values()), dtype=np.float32)
            blue_data.append((player_dict, arr))
        
        # Do the same for the red team
        red_data = []
        for player in red_players:
            player_dict = dict(player)
            arr = np.array(list(player_dict.values()), dtype=np.float32)
            red_data.append((player_dict, arr))
        
        # Compute team sums in one go using NumPy
        blue_sum = np.sum([arr for _, arr in blue_data], axis=0)
        red_sum = np.sum([arr for _, arr in red_data], axis=0)
        all_sum = blue_sum + red_sum
        
        # Process each player's frame data
        for idx, (player_dict, arr) in enumerate(blue_data + red_data):
            team_sum = blue_sum if idx < 5 else red_sum
            
            # Compute percentages with proper handling for division by zero
            player_team_percent = np.divide(arr, team_sum, out=np.zeros_like(arr), where=team_sum != 0)
            player_all_percent = np.divide(arr, all_sum, out=np.zeros_like(arr), where=all_sum != 0)
            
            # Enrich the player's dictionary with percentage values
            enriched_frame = {}
            for i, key in enumerate(player_dict):
                enriched_frame[key] = player_dict[key]
                enriched_frame[f"{key}_team_percent"] = player_team_percent[i]
                enriched_frame[f"{key}_all_percent"] = player_all_percent[i]
            
            player_frames[idx].append(enriched_frame)

    return player_frames


if __name__ == "__main__":
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client["embedded-rift"]
    collection = db["games"]

    game_doc = collection.find_one({"$expr": {"$gte": [{"$divide": ["$result.gameDuration", 60]}, 26]}})
    frames_data = extract_frames(game_doc)
    players_data = extract_all_players_frames(frames_data)

    print(players_data.keys())
    print(players_data[0][0])
    to_json = { 
        player_idx: [{k: float(val) for k, val in frame.items()} for frame in frames] 
        for player_idx, frames in players_data.items()
    }
    for player_idx, frames_list in players_data.items():
        team = "blue" if player_idx < 5 else "red"
        sample_frame = frames_list[20]  # Access the 21st frame (index 20)
        print(sample_frame.get(f"{team}_{player_idx % 5}_kills"))
        print(sample_frame.get(f"{team}_{player_idx % 5}_deaths"))
        # Create a numpy array from frame data values for the first 10 keys (assuming consistent key ordering)
        pi = np.array([[float(d.get(key, 0)) for key in d] for d in frames_list], dtype=np.float32)
        print(pi[20, :10])
        
    with open("frames.json", "w") as f:
        json.dump(to_json, f, indent=2)
