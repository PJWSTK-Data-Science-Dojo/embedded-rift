from typing import List, Dict, Any, Optional
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import logging
import json
from dotenv import load_dotenv
import os
from utils.riot_api import RiotAPI, Platform, Region
import random
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.collection import Collection
import argparse


def get_high_elo_matches(api: RiotAPI, platform: Platform, region: Region) -> List[str]:
    """
    Fetches HighElo players and returns a list of match IDs.
    """
    print("Fetching HighElo players...")
    high_elo_players: List[Dict[str, Any]] = api.get_apex_tiers_summoner_ids(platform)

    if not high_elo_players:
        print("No HighElo players found.")
        return []

    print(f"Found {len(high_elo_players)} HighElo players.")

    with open("data/high_elo_players.json", "w") as f:
        json.dump(high_elo_players, f, indent=4)

    # Shuffle players to ensure randomness
    random.shuffle(high_elo_players)

    match_ids: set[str] = set()
    start_time = datetime.now() - timedelta(days=7)

    for player in tqdm(high_elo_players, desc="Fetching Match IDs"):
        summoner_id: str = player["summonerId"]
        puuid: Optional[str] = api.get_puuid_by_summoner_id(platform, summoner_id)

        if not puuid:
            print(f"\n\nFailed to get PUUID for summoner ID: {summoner_id}\n")
            continue

        matches: List[str] = api.get_player_matches_ids(
            region, puuid, start_time=start_time, count=10
        )
        if matches:
            match_ids.update(matches)

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name: str = f"data/{platform}-{timestamp}-{len(match_ids)}-matches-ids.json"

    with open(file_name, "w") as f:
        json.dump(list(match_ids), f, indent=4)
    print(f"Saved final {len(match_ids)} matches to {file_name}.")

    return list(match_ids)


# Configure logging
logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ConnectionError),
)
def fetch_with_retries(api: RiotAPI, region: Region, match_id: str) -> Dict[str, Any]:
    return api.get_match_data(region, match_id)


def fetch_and_save_data(
    api: RiotAPI,
    region: Region,
    match_ids: List[str],
    collection: Collection,
    max_attempts: int = 3,
) -> None:
    """
    Iterates through match_ids, fetching match data and saving it to MongoDB.
    If fetching fails (whether due to ConnectionError or another Exception),
    the match ID is saved to a corresponding failed list. At the end, these
    failed match IDs are written to JSON files.
    """
    failed_connection_ids: List[str] = []
    failed_exception_ids: List[str] = []

    for match_id in tqdm(match_ids, desc="Fetching Matches Data"):
        try:
            match = fetch_with_retries(api, region, match_id)
            collection.insert_one(match)
        except ConnectionError as ce:
            logging.error(f"ConnectionError for match {match_id}: {ce}")
            failed_connection_ids.append(match_id)
        except Exception as e:
            logging.error(f"Failed to fetch match {match_id}: {e}")
            failed_exception_ids.append(match_id)

    # Save failed match IDs to separate JSON files
    with open("data/failed_connection_matches.json", "w") as f:
        json.dump(failed_connection_ids, f, indent=4)
    with open("data/failed_exception_matches.json", "w") as f:
        json.dump(failed_exception_ids, f, indent=4)

    print(f"Failed (ConnectionError) matches: {failed_connection_ids}")
    print(f"Failed (Other Exception) matches: {failed_exception_ids}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High Elo Matches fetcher to MongoDB")
    parser.add_argument(
        "-p",
        "--platform",
        type=Platform,  # Convert argument to the corresponding Enum member
        choices=list(Platform),  # Restrict choices to enum members
        default=Platform.EUW,
        help="Platform to fetch data from",
    )
    parser.add_argument(
        "-r",
        "--region",
        type=Region,  # Convert argument to the corresponding Enum member
        choices=list(Region),  # Restrict choices to enum members
        default=Region.EUROPE,
        help="Region to fetch data from",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    API_KEY: Optional[str] = os.getenv("RIOT_API")
    MONGO_URI: Optional[str] = os.getenv("MONGO_URI")

    args = parse_args()
    PLATFORM: Platform = args.platform
    REGION: Region = args.region

    client = MongoClient(MONGO_URI)
    db = client["embedded-rift"]
    collection: Collection = db["games"]

    riot_api = RiotAPI(api_key=API_KEY)

    random_matches: List[str] = get_high_elo_matches(riot_api, PLATFORM, REGION)

    fetch_and_save_data(riot_api, REGION, random_matches, collection)
