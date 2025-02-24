from typing import Any, Dict, List, Tuple
from dataclasses import asdict, dataclass
import numpy as np
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import json

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class PlayerFrame:
    kills: int
    deaths: int
    assists: int
    # skills: List[int]
    turretPlatesDestroyed: int
    wardsPlaced: int
    wardsDestroyed: int

    # stats
    abilityHaste: int
    abilityPower: int
    armor: int
    armorPen: int
    armorPenPercent: int
    attackDamage: int
    attackSpeed: int
    bonusArmorPenPercent: int
    bonusMagicPenPercent: int
    ccReduction: int
    cooldownReduction: int
    # health: int
    healthMax: int
    healthRegen: int
    lifesteal: int
    magicPen: int
    magicPenPercent: int
    magicResist: int
    movementSpeed: int
    omnivamp: int
    physicalVamp: int
    # power: int
    # powerMax: int
    # powerRegen: int
    spellVamp: int

    jungleMinionsKilled: int
    minionsKilled: int

    totalGold: int
    currentGold: int
    goldPerSecond: int

    level: int
    xp: int

    # damageStats
    magicDamageDone: int
    magicDamageDoneToChampions: int
    magicDamageTaken: int
    physicalDamageDone: int
    physicalDamageDoneToChampions: int
    physicalDamageTaken: int
    totalDamageDone: int
    totalDamageDoneToChampions: int
    totalDamageTaken: int
    trueDamageDone: int
    trueDamageDoneToChampions: int
    trueDamageTaken: int

    # special features
    visionScore: int  # Scales time * sqrt(time)
    totalHeal: int  # Scales time * sqrt(time)
    totalDamageShieldedOnTeammates: int  # Scales quadratically with time
    totalDamageToBuildings: int  # Scales logarithmically with time from 7 min to EOG
    totalDamageToObjectives: int  # Scales linearly with time from 5 min to EOG
    selfMitigatedDamage: int  # Scales quadratically with time

    # participantId: int
    # x: int
    # y: int
    timeEnemySpentControlled: int


@dataclass
class GameObjectives:
    voidGrub: int
    riftHerald: int
    baronNashor: int
    atakhan: int
    drake: int
    elderDrake: int

    dragonSoul: int

    # buildings
    turret: int
    inhibitor: int


@dataclass
class Team:
    objectives: GameObjectives
    players: List[PlayerFrame]


@dataclass
class FrameInput:
    blue: Team
    red: Team


@dataclass
class GameInput:
    game_id: str
    frames: List[Dict[str, Any]]  # Flatten
    blue_champions: List[int]
    red_champions: List[int]
    items_per_frame: List[List[int]]
    game_duration: int
    early_surrender: bool
    surrender: bool
    blue_win: bool
    platform: str
    season: str
    patch: str


def xsqrtx(x: float, game_duration: float) -> float:
    return x * np.sqrt(x) / (game_duration * np.sqrt(game_duration))


def quadratic(x: float, game_duration: float) -> float:
    return x * x / (game_duration * game_duration)


def logarithmic(x: float, game_duration: float) -> float:
    a = 1.25
    t0 = 10
    # print(game_duration)
    if game_duration < t0:
        return x

    val = np.log(a * (t0 - 1) / t0) / np.log(a * game_duration / t0)
    a2 = val / ((t0) ** 2)
    return np.where(
        x <= t0,
        a2 * x**2,
        np.log(a * x / t0) / np.log((a * game_duration / t0)),
    )


def linear(x: float, game_duration: float, b: float = 0) -> float:
    return max((x - b) / (game_duration - b), 0)


def extract_player(
    p: Dict[str, Any],
    player_result: Dict[str, int],
    frame_index: int,
    game_duration: float,
) -> Tuple[PlayerFrame, List[int]]:
    """
    Extracts team-level data (objectives and player frames) from the team_data dictionary.
    """

    visionScore = xsqrtx(frame_index, game_duration) * player_result.get(
        "visionScore", 0
    )
    totalHeal = xsqrtx(frame_index, game_duration) * player_result.get("totalHeal", 0)
    totalDamageShieldedOnTeammates = quadratic(
        frame_index, game_duration
    ) * player_result.get("totalDamageShieldedOnTeammates", 0)
    totalDamageToBuildings = logarithmic(
        frame_index, game_duration
    ) * player_result.get("damageDealtToBuildings", 0)

    totalDamageToObjectives = linear(frame_index, game_duration, 5) * player_result.get(
        "damageDealtToObjectives", 0
    )
    selfMitigatedDamage = quadratic(frame_index, game_duration) * player_result.get(
        "damageSelfMitigated", 0
    )

    ed: Dict[str, int] = p.get("eventData", {})
    items = ed.get("items", [0, 0, 0, 0, 0, 0])
    cs: Dict[str, int] = p.get("championStats", {})
    ds: Dict[str, int] = p.get("damageStats", {})

    return (
        PlayerFrame(
            kills=ed.get("kills", 0),
            deaths=ed.get("deaths", 0),
            assists=ed.get("assists", 0),
            turretPlatesDestroyed=ed.get("turretPlatesDestroyed", 0),
            wardsPlaced=ed.get("wardsPlaced", 0),
            wardsDestroyed=ed.get("wardsDestroyed", 0),
            abilityHaste=cs.get("abilityHaste", 0),
            abilityPower=cs.get("abilityPower", 0),
            armor=cs.get("armor", 0),
            armorPen=cs.get("armorPen", 0),
            armorPenPercent=cs.get("armorPenPercent", 0),
            attackDamage=cs.get("attackDamage", 0),
            attackSpeed=cs.get("attackSpeed", 0),
            bonusArmorPenPercent=cs.get("bonusArmorPenPercent", 0),
            bonusMagicPenPercent=cs.get("bonusMagicPenPercent", 0),
            ccReduction=cs.get("ccReduction", 0),
            cooldownReduction=cs.get("cooldownReduction", 0),
            healthMax=cs.get("healthMax", 0),
            healthRegen=cs.get("healthRegen", 0),
            lifesteal=cs.get("lifesteal", 0),
            magicPen=cs.get("magicPen", 0),
            magicPenPercent=cs.get("magicPenPercent", 0),
            magicResist=cs.get("magicResist", 0),
            movementSpeed=cs.get("movementSpeed", 0),
            omnivamp=cs.get("omnivamp", 0),
            physicalVamp=cs.get("physicalVamp", 0),
            spellVamp=cs.get("spellVamp", 0),
            jungleMinionsKilled=p.get("jungleMinionsKilled", 0),
            minionsKilled=p.get("minionsKilled", 0),
            totalGold=p.get("totalGold", 0),
            currentGold=p.get("currentGold", 0),
            goldPerSecond=p.get("goldPerSecond", 0),
            level=p.get("level", 0),
            xp=p.get("xp", 0),
            magicDamageDone=ds.get("magicDamageDone", 0),
            magicDamageDoneToChampions=ds.get("magicDamageDoneToChampions", 0),
            magicDamageTaken=ds.get("magicDamageTaken", 0),
            physicalDamageDone=ds.get("physicalDamageDone", 0),
            physicalDamageDoneToChampions=ds.get("physicalDamageDoneToChampions", 0),
            physicalDamageTaken=ds.get("physicalDamageTaken", 0),
            totalDamageDone=ds.get("totalDamageDone", 0),
            totalDamageDoneToChampions=ds.get("totalDamageDoneToChampions", 0),
            totalDamageTaken=ds.get("totalDamageTaken", 0),
            trueDamageDone=ds.get("trueDamageDone", 0),
            trueDamageDoneToChampions=ds.get("trueDamageDoneToChampions", 0),
            trueDamageTaken=ds.get("trueDamageTaken", 0),
            visionScore=visionScore,
            totalHeal=totalHeal,
            totalDamageShieldedOnTeammates=totalDamageShieldedOnTeammates,
            totalDamageToBuildings=totalDamageToBuildings,
            totalDamageToObjectives=totalDamageToObjectives,
            selfMitigatedDamage=selfMitigatedDamage,
            timeEnemySpentControlled=p.get("timeEnemySpentControlled", 0),
        ),
        items,
    )


def extract_team(
    team_data: Dict[str, Any],
    team_result: Dict[int, Dict[str, int]],
    frame_index: int,
    game_duration: float,
) -> Tuple[Team, List[int]]:
    """
    Extracts a Team object from team_data, interpolating missing player fields.
    final_stats_map maps participantId to a dictionary of final post-game values.
    """
    team_event_data = team_data.get("eventData", {})
    objectives = GameObjectives(
        voidGrub=team_event_data.get("eliteMonstersKilled", {}).get("voidGrub", 0),
        riftHerald=team_event_data.get("eliteMonstersKilled", {}).get("riftHerald", 0),
        baronNashor=team_event_data.get("eliteMonstersKilled", {}).get(
            "baronNashor", 0
        ),
        atakhan=team_event_data.get("eliteMonstersKilled", {}).get("atakhan", 0),
        drake=team_event_data.get("eliteMonstersKilled", {}).get("drake", 0),
        elderDrake=team_event_data.get("eliteMonstersKilled", {}).get("elderDrake", 0),
        dragonSoul=team_event_data.get("dragonSoul", 0),
        turret=team_event_data.get("buildingsDestroyed", {}).get("turret", 0),
        inhibitor=team_event_data.get("buildingsDestroyed", {}).get("inhibitor", 0),
    )
    # print(team_result)
    players = []
    team_items = []
    for i, p in enumerate(team_data["participants"]):
        # print(participant_id)
        # Get the final post-game stats for this participant.
        player_reult = team_result["participants"][i]
        player, items = extract_player(p, player_reult, frame_index, game_duration)
        players.append(player)
        team_items.extend(items)

    return Team(objectives=objectives, players=players), team_items


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Dict[str, Any]:
    """
    Recursively flattens a nested dictionary.
    For lists, each element is flattened with its index appended.
    """
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_dict(v, new_key, sep=sep))
    elif isinstance(d, list):
        for idx, item in enumerate(d):
            new_key = f"{parent_key}{sep}{idx}"
            items.update(flatten_dict(item, new_key, sep=sep))
    else:
        items[parent_key] = d
    return items


def extract_game_data(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts timeline frames from the game_data and converts each frame into a FrameInput.
    """
    game_id = game_data["metadata"]["matchId"]
    platform = game_data["metadata"]["platform"]
    season = game_data["metadata"]["season"]
    patch = game_data["metadata"]["patch"]

    timeline = game_data["timeline"]
    frame_sequence: List[FrameInput] = []
    result = game_data.get("result", {})

    if isinstance(result["teams"], list):
        blue_result = result["teams"][0]
        red_result = result["teams"][1]
    else:
        blue_result = result["teams"].get("blue", {})
        red_result = result["teams"].get("red", {})

    blue_early_surrender = blue_result["participants"][0].get(
        "gameEndedInEarlySurrender", False
    )
    red_early_surrender = red_result["participants"][0].get(
        "gameEndedInEarlySurrender", False
    )

    early_surrender = blue_early_surrender or red_early_surrender

    blue_surender = blue_result["participants"][0].get("gameEndedInSurrender", False)

    red_surrender = red_result["participants"][0].get("gameEndedInSurrender", False)
    surrender = blue_surender or red_surrender

    # Get game duration from the result section (assume in seconds or minutes consistently)
    game_duration = game_data.get("result", {}).get("gameDuration", 0)

    if game_duration < 60:
        return []

    raw_duration = game_duration
    game_duration = np.ceil(game_duration / 60)
    items_per_frame = []
    blue_champions = []

    for player in blue_result["participants"]:
        blue_champions.append(player["championId"])

    red_champions = []
    for player in red_result["participants"]:
        red_champions.append(player["championId"])

    # Assume timeline frames are evenly spaced. Compute elapsed time for each frame.
    for idx, frame in enumerate(timeline):

        teams_frame = frame.get("teams", {})

        blue_frame = teams_frame.get("blue", {})

        if not blue_frame:
            blue_frame = teams_frame.get(0, {})

        items_in_frame = []
        blue_team, blue_items = extract_team(
            blue_frame,
            blue_result,
            idx,
            game_duration,
        )
        items_in_frame.extend(blue_items)

        red_frame = teams_frame.get("red", {})
        if not red_frame:
            red_frame = teams_frame.get(1, {})

        red_team, red_items = extract_team(
            red_frame,
            red_result,
            idx,
            game_duration,
        )
        items_in_frame.extend(red_items)

        items_per_frame.append(items_in_frame)
        frame_input = FrameInput(blue=blue_team, red=red_team)
        frame_sequence.append(flatten_dict(asdict(frame_input)))

    game_input = GameInput(
        game_id=game_id,
        frames=frame_sequence,
        game_duration=raw_duration,
        items_per_frame=items_per_frame,
        blue_champions=blue_champions,
        red_champions=red_champions,
        early_surrender=early_surrender,
        surrender=surrender,
        blue_win=blue_result.get("win", False),
        platform=platform,
        season=season,
        patch=patch,
    )
    return asdict(game_input)


if __name__ == "__main__":
    # Example usage\
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    # DB_NAME = os.getenv("DB_NAME")
    client = MongoClient(MONGO_URI)
    db = client["embedded-rift"]
    collection = db["games"]

    game_data = collection.find_one(
        {"$expr": {"$gte": [{"$divide": ["$result.gameDuration", 60]}, 26]}}
    )

    game_data = extract_game_data(game_data)
    # frames = [flatten_dict(frame) for frame in game_data["frames"]]

    with open("frames.json", "w") as f:
        json.dump(game_data, f, indent=1)
