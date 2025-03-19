

from itertools import islice
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient

from utils.training.frames import extract_frames


def extract_team_objectives(game_input: Dict[str, Any]):
    """
    Extracts the team objectives from the game input.
    """
    team_objectives = {
        "blue": [],
        "red": []
    }
    
    
    frames: List[Dict[str, Any]] = game_input["frames"]
    for frame in frames:
        keys = list(frame.keys())
        l = len(keys)//2
        blue, red = list(islice(frame.items(), l)), list(islice(frame.items(), l, None))
        
        blue_obj, blue_players = dict(blue[:9]), blue[9:]
        red_obj, red_players = dict(red[:9]), red[9:]
        blue_obj["blue_objectives_dragonSoul"] = 1 if blue_obj["blue_objectives_dragonSoul"] else 0
        red_obj["red_objectives_dragonSoul"] = 1 if red_obj["red_objectives_dragonSoul"] else 0
        
        team_objectives["blue"].append({**blue_obj})
        team_objectives["red"].append({**red_obj})
        
    return team_objectives


if __name__ == "__main__":
    # Example usage\
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    # DB_NAME = os.getenv("DB_NAME")
    client = MongoClient(MONGO_URI)
    db = client["embedded-rift"]
    collection = db["games"]

    frames = collection.find_one(
        {"$expr": {"$gte": [{"$divide": ["$result.gameDuration", 60]}, 26]}}
    )

    frames = extract_frames(frames)
    # frames = [flatten_dict(frame) for frame in game_data["frames"]]
    data = extract_team_objectives(frames)
    with open("frames.json", "w") as f:
        json.dump(data, f, indent=2)
