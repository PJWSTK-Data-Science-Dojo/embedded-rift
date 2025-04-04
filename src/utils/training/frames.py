import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

a = 1.25
t0 = 10
log_divisor = 1
a2 = np.log(a * (t0 - 1) / t0) / log_divisor / (t0 ** 2)
sqrt_denom = 1
quad_denom = 1

ALL_ROLES = {
    "TOP",
    "JUNGLE",
    "MIDDLE",
    "BOTTOM",
    "UTILITY",
}

def xsqrtx(x: float) -> float:
    """Scales x with a square root function using a precomputed denominator."""
    global sqrt_denom
    return x * np.sqrt(x) / sqrt_denom


def quadratic(x: float) -> float:
    """Scales x quadratically using a precomputed denominator."""
    global quad_denom
    return (x * x) / quad_denom


def logarithmic(x: float, game_duration: float) -> float:
    """Scales x logarithmically using precomputed constants."""
    global a, t0, log_divisor, a2
    if game_duration < t0:
        return x

    return np.where(x <= t0, a2 * x ** 2, np.log(a * x / t0) / log_divisor)


def linear(x: float, game_duration: float, b: float = 0) -> float:
    """Scales x linearly above a threshold."""
    return max((x - b) / (game_duration - b), 0)


def extract_player(
    p: Dict[str, Any],
    player_result: Dict[str, int],
    frame_index: int,
    game_duration: float,
) -> Tuple[Dict[str, Any], List[int]]:
    """
    Extracts a player's frame data as a dictionary and returns it along with the item list.
    Uses precomputed denominators and logarithmic constants.
    """

    # Precompute scaled metrics
    visionScore = xsqrtx(frame_index) * player_result.get("visionScore", 0)
    totalHeal = xsqrtx(frame_index) * player_result.get("totalHeal", 0)
    totalDamageShieldedOnTeammates = quadratic(frame_index) * player_result.get("totalDamageShieldedOnTeammates", 0)
    totalDamageToBuildings = logarithmic(frame_index, game_duration=game_duration) * player_result.get("damageDealtToBuildings", 0)
    totalDamageToObjectives = linear(frame_index, game_duration, 5) * player_result.get("damageDealtToObjectives", 0)
    selfMitigatedDamage = quadratic(frame_index) * player_result.get("damageSelfMitigated", 0)

    # Cache nested dictionary lookups
    ed = p.get("eventData", {})
    cs = p.get("championStats", {})
    ds = p.get("damageStats", {})

    player_dict = {
        "kills": ed.get("kills", 0),
        "deaths": ed.get("deaths", 0),
        "assists": ed.get("assists", 0),
        "turretPlatesDestroyed": ed.get("turretPlatesDestroyed", 0),
        "wardsPlaced": ed.get("wardsPlaced", 0),
        "wardsDestroyed": ed.get("wardsDestroyed", 0),
        "abilityHaste": cs.get("abilityHaste", 0),
        "abilityPower": cs.get("abilityPower", 0),
        "armor": cs.get("armor", 0),
        "armorPen": cs.get("armorPen", 0),
        "armorPenPercent": cs.get("armorPenPercent", 0),
        "attackDamage": cs.get("attackDamage", 0),
        "attackSpeed": cs.get("attackSpeed", 0),
        "bonusArmorPenPercent": cs.get("bonusArmorPenPercent", 0),
        "bonusMagicPenPercent": cs.get("bonusMagicPenPercent", 0),
        "ccReduction": cs.get("ccReduction", 0),
        "cooldownReduction": cs.get("cooldownReduction", 0),
        "healthMax": cs.get("healthMax", 0),
        "healthRegen": cs.get("healthRegen", 0),
        "lifesteal": cs.get("lifesteal", 0),
        "magicPen": cs.get("magicPen", 0),
        "magicPenPercent": cs.get("magicPenPercent", 0),
        "magicResist": cs.get("magicResist", 0),
        "movementSpeed": cs.get("movementSpeed", 0),
        "omnivamp": cs.get("omnivamp", 0),
        "physicalVamp": cs.get("physicalVamp", 0),
        "spellVamp": cs.get("spellVamp", 0),
        "jungleMinionsKilled": p.get("jungleMinionsKilled", 0),
        "minionsKilled": p.get("minionsKilled", 0),
        "totalGold": p.get("totalGold", 0),
        "currentGold": p.get("currentGold", 0),
        "goldPerSecond": p.get("goldPerSecond", 0),
        "level": p.get("level", 0),
        "xp": p.get("xp", 0),
        "magicDamageDone": ds.get("magicDamageDone", 0),
        "magicDamageDoneToChampions": ds.get("magicDamageDoneToChampions", 0),
        "magicDamageTaken": ds.get("magicDamageTaken", 0),
        "physicalDamageDone": ds.get("physicalDamageDone", 0),
        "physicalDamageDoneToChampions": ds.get("physicalDamageDoneToChampions", 0),
        "physicalDamageTaken": ds.get("physicalDamageTaken", 0),
        "totalDamageDone": ds.get("totalDamageDone", 0),
        "totalDamageDoneToChampions": ds.get("totalDamageDoneToChampions", 0),
        "totalDamageTaken": ds.get("totalDamageTaken", 0),
        "trueDamageDone": ds.get("trueDamageDone", 0),
        "trueDamageDoneToChampions": ds.get("trueDamageDoneToChampions", 0),
        "trueDamageTaken": ds.get("trueDamageTaken", 0),
        "visionScore": visionScore,
        "totalHeal": totalHeal,
        "totalDamageShieldedOnTeammates": totalDamageShieldedOnTeammates,
        "totalDamageToBuildings": totalDamageToBuildings,
        "totalDamageToObjectives": totalDamageToObjectives,
        "selfMitigatedDamage": selfMitigatedDamage,
        "timeEnemySpentControlled": p.get("timeEnemySpentControlled", 0)
    }
    # Items list from eventData
    items = ed.get("items", [0, 0, 0, 0, 0, 0])
    return player_dict, items


def extract_team(
    team_data: Dict[str, Any],
    team_result: Dict[str, Any],
    frame_index: int,
    game_duration: float,
) -> Dict[str, Any]:
    """
    Extracts a team's data as a dictionary with keys: 'objectives', 'players', and 'items'.
    """
    ted = team_data.get("eventData", {})
    objectives = {
        "voidGrub": ted.get("eliteMonstersKilled", {}).get("voidGrub", 0),
        "riftHerald": ted.get("eliteMonstersKilled", {}).get("riftHerald", 0),
        "baronNashor": ted.get("eliteMonstersKilled", {}).get("baronNashor", 0),
        "atakhan": ted.get("eliteMonstersKilled", {}).get("atakhan", 0),
        "drake": ted.get("eliteMonstersKilled", {}).get("drake", 0),
        "elderDrake": ted.get("eliteMonstersKilled", {}).get("elderDrake", 0),
        "dragonSoul": ted.get("dragonSoul", 0),
        "turret": ted.get("buildingsDestroyed", {}).get("turret", 0),
        "inhibitor": ted.get("buildingsDestroyed", {}).get("inhibitor", 0),
    }
    players = []
    team_items = []
    for i, p in enumerate(team_data.get("participants", [])):
        player_result = team_result["participants"][i]
        player, items = extract_player(p, player_result, frame_index, game_duration )
        players.append(player)
        team_items.append(items)
    return {"objectives": objectives, "players": players, "items": team_items}


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """
    Iteratively flattens a nested dictionary (including lists) into a single-level dictionary.
    This version preserves the order of keys as they appear in the original inner dictionaries.
    """
    items = {}
    stack = [(parent_key, d)]
    while stack:
        cur_key, cur_val = stack.pop()
        if isinstance(cur_val, dict):
            # Reverse the order when pushing so that the first key is processed first.
            for k, v in reversed(list(cur_val.items())):
                new_key = f"{cur_key}{sep}{k}" if cur_key else k
                stack.append((new_key, v))
        elif isinstance(cur_val, list):
            # For lists, reverse the order of indices to preserve the original order.
            for idx, item in reversed(list(enumerate(cur_val))):
                new_key = f"{cur_key}{sep}{idx}" if cur_key else str(idx)
                stack.append((new_key, item))
        else:
            items[cur_key] = cur_val
    return items

def extract_frames(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts timeline frames from game_data into a flattened dictionary structure.
    Uses plain dicts instead of dataclasses.
    """
    global log_divisor, a2, sqrt_denom, quad_denom
    
    metadata = game_data["metadata"]
    game_id = metadata["matchId"]
    platform = metadata["platform"]
    season = metadata["season"]
    patch = metadata["patch"]

    timeline = game_data["timeline"]
    frame_sequence = []
    result = game_data.get("result", {})

    # Determine blue and red team results
    if isinstance(result.get("teams"), list):
        blue_result, red_result = result["teams"][0], result["teams"][1]
    else:
        teams = result.get("teams", {})
        blue_result, red_result = teams.get("blue", {}), teams.get("red", {})

    blue_early = blue_result["participants"][0].get("gameEndedInEarlySurrender", False)
    red_early = red_result["participants"][0].get("gameEndedInEarlySurrender", False)
    early_surrender = blue_early or red_early

    blue_surr = blue_result["participants"][0].get("gameEndedInSurrender", False)
    red_surr = red_result["participants"][0].get("gameEndedInSurrender", False)
    surrender = blue_surr or red_surr

    raw_duration = result.get("gameDuration", 0)
    if raw_duration < 60:
        return {}
    # Convert duration to minutes (ceiling)
    game_duration = np.ceil(raw_duration / 60)

    # Precompute constants (only once per game)
    log_divisor = np.log(a * game_duration / t0)
    sqrt_denom = game_duration * np.sqrt(game_duration)
    quad_denom = game_duration * game_duration

    blue_champions = [p["championId"] for p in blue_result.get("participants", [])]
    red_champions = [p["championId"] for p in red_result.get("participants", [])]
    blue_positions = [p["teamPosition"] for p in blue_result.get("participants", [])]
    red_positions = [p["teamPosition"] for p in red_result.get("participants", [])]
    if any(pos not in ALL_ROLES for pos in blue_positions + red_positions):
        raise ValueError("Invalid team positions found in game data.")
    
    items_per_frame = []
    objectives_list = []

    for idx, frame in enumerate(timeline):
        teams_frame = frame.get("teams", {})
        blue_frame = teams_frame.get("blue", teams_frame.get(0, {}))
        red_frame = teams_frame.get("red", teams_frame.get(1, {}))

        objectives_in_frame = []
        items_in_frame = []

        blue_team = extract_team(blue_frame, blue_result, idx, game_duration)
        red_team = extract_team(red_frame, red_result, idx, game_duration)

        items_in_frame.extend(blue_team["items"])
        items_in_frame.extend(red_team["items"])
        objectives_in_frame.append(blue_team["objectives"])
        objectives_in_frame.append(red_team["objectives"])

        objectives_list.append(objectives_in_frame)
        items_per_frame.append(items_in_frame)

        frame_input = {"blue": blue_team["players"], "red": red_team["players"]}
        flat_frame = flatten_dict(frame_input)
        frame_sequence.append(flat_frame)

    game_input = {
        "game_id": game_id,
        "blue_champions": blue_champions,
        "red_champions": red_champions,
        "frames": frame_sequence,
        "game_duration": raw_duration,
        "items_per_frame": items_per_frame,
        "objectives": objectives_list,
        "early_surrender": early_surrender,
        "surrender": surrender,
        "blue_win": blue_result.get("win", False),
        "blue_positions": blue_positions,
        "red_positions": red_positions,
        "platform": platform,
        "season": season,
        "patch": patch,
    }
    return game_input


if __name__ == "__main__":
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client["embedded-rift"]
    collection = db["games"]

    game_document = collection.find_one({"$expr": {"$gte": [{"$divide": ["$result.gameDuration", 60]}, 26]}})
    extracted_frames = extract_frames(game_document)

    with open("frames.json", "w") as f:
        json.dump(extracted_frames, f, indent=1)
