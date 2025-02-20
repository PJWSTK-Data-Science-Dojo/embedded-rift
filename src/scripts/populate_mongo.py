from typing import List, Dict, Any, Optional, Set
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
from pymongo.errors import ConnectionError, DuplicateKeyError
from pymongo.collection import Collection
import argparse


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ConnectionError),
)
def fetch_matches_with_retries(
    api: RiotAPI,
    summoner_id: str,
    region: Region,
    platform: Platform,
    start_time: datetime,
) -> Optional[List[str]]:
    """
    For a given summoner, fetch match IDs.
    Returns a list of match IDs or None if puuid could not be obtained.
    """
    puuid: Optional[str] = api.get_puuid_by_summoner_id(platform, summoner_id)
    if not puuid:
        print(f"Failed to get PUUID for summoner ID: {summoner_id}")
        return None

    matches: List[str] = api.get_player_matches_ids(
        region, puuid, start_time=start_time, count=10
    )
    return matches


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ConnectionError),
)
def fetch_with_retries(api: RiotAPI, region: Region, match_id: str) -> Dict[str, Any]:
    """
    Fetches match data for a given match_id.
    """
    return api.get_match_data(region, match_id)


def fetch_and_save_player_matches(
    api: RiotAPI,
    platform: Platform,
    region: Region,
    collection: Collection,
) -> None:
    """
    For each high-elo player:
      - Fetch match IDs.
      - For each match ID not already in MongoDB, fetch the match data and insert it.
    Maintains a set of existing match IDs (based on metadata.matchId).
    Logs and saves IDs of failed fetches.
    """
    print("Fetching HighElo players...")
    high_elo_players: List[Dict[str, Any]] = api.get_apex_tiers_summoner_ids(platform)
    if not high_elo_players:
        print("No HighElo players found.")
        return

    print(f"Found {len(high_elo_players)} HighElo players.")

    # Save players for reference.
    with open("data/high_elo_players.json", "w") as f:
        json.dump(high_elo_players, f, indent=4)

    # Shuffle players for randomness.
    random.shuffle(high_elo_players)
    start_time = datetime.now() - timedelta(days=7)

    # Build a set of already-fetched match IDs from the collection.
    existing_matches: Set[str] = set()
    for doc in tqdm(
        collection.find({}, {"metadata.matchId": 1}), desc="Fetching existing matches"
    ):
        metadata = doc.get("metadata", {})
        match_id = metadata.get("matchId")
        if match_id:
            existing_matches.add(match_id)

    print(f"Found {len(existing_matches)} existing matches in the database.")

    failed_connection_ids: List[str] = []
    failed_exception_ids: List[str] = []

    # Process each player.
    for player in tqdm(high_elo_players, desc="Processing players"):
        summoner_id: str = player["summonerId"]
        try:
            match_ids: Optional[List[str]] = fetch_matches_with_retries(
                api, summoner_id, region, platform, start_time
            )
        except Exception as e:
            logging.error(f"Error fetching matches for summoner ID {summoner_id}: {e}")
            continue

        if not match_ids:
            continue

        # For each match id, check if it already exists.
        for match_id in match_ids:
            if match_id in existing_matches:
                print(f"Skipping match {match_id} (already in DB)")
                continue

            try:
                match: Dict[str, Any] = fetch_with_retries(api, region, match_id)
                # Optionally, set the document _id as the match id from metadata.
                if match.get("metadata", {}).get("matchId"):
                    match["_id"] = match["metadata"]["matchId"]
                existing_matches.add(match["metadata"]["matchId"])
                collection.insert_one(match)
            except ConnectionError as ce:
                logging.error(f"ConnectionError for match {match_id}: {ce}")
                failed_connection_ids.append(match_id)
            except DuplicateKeyError as dke:
                pass  # Ignore duplicate key errors.
            except Exception as e:
                logging.error(f"Failed to fetch match {match_id}: {e}")
                failed_exception_ids.append(match_id)

    # Save failed match IDs to separate JSON files.
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
        type=Platform,
        choices=list(Platform),
        default=Platform.EUW,
        help="Platform to fetch data from",
    )
    parser.add_argument(
        "-r",
        "--region",
        type=Region,
        choices=list(Region),
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

    fetch_and_save_player_matches(riot_api, PLATFORM, REGION, collection)
