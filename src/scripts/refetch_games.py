import argparse
import logging
from typing import Any, Dict, List
import h5py
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from utils.riot_api import Platform, Region, RiotAPI
from tqdm import tqdm
import json
from dotenv import load_dotenv
import os
from collections import deque
from pymongo import MongoClient
from pymongo.collection import Collection


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


def get_games_ids_to_refetch(hdf5_filename: str, group_name: str) -> List[str]:
    """
    Returns a list of game IDs that have no frames stored in the HDF5 file.
    """
    game_ids = []
    with h5py.File(hdf5_filename, "r") as hf:
        games_group = hf[group_name]
        for game_id in tqdm(games_group, "Filtering game IDs"):
            game_ds = games_group[game_id]
            game_duration = game_ds.attrs.get("game_duration", 0) / 60  # in minutes

            if game_duration < 10:
                continue
            game_ids.append(game_id)

    return game_ids


def split_game_ids(game_ids: List[str]) -> dict[Region, List[str]]:
    result = {
        Region.ASIA: [],
        Region.EUROPE: [],
    }
    for game_id in tqdm(game_ids, "Splitting game IDs"):
        if game_id.startswith("KR"):
            result[Region.ASIA].append(game_id)
        else:
            result[Region.EUROPE].append(game_id)
    return result


def refetch_games(
    game_ids: List[str], api: RiotAPI, region: Region, collection: Collection
) -> None:
    games = deque(game_ids)

    with tqdm(game_ids, desc="Refetching games") as pbar:
        while games:
            game_id = games.popleft()
            try:
                game_data = fetch_with_retries(api, region, game_id)
                collection.insert_one(game_data)
                pbar.update(1)
            except Exception as e:
                logging.error(f"Failed to fetch game data for game ID: {game_id}.")
                games.append(game_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refetcher to MongoDB")

    parser.add_argument(
        "-r",
        "--region",
        type=Region,
        choices=list(Region),
        default=Region.EUROPE,
        help="Region to fetch data from",
    )

    return parser.parse_args()


def main():
    load_dotenv()
    hdf5_filename = "games_data.h5"
    group_name = "games"
    RIOT_API_KEY = os.getenv("RIOT_API")
    MONGO_URI = os.getenv("MONGO_URI")

    args = parse_args()
    region = args.region

    game_ids = get_games_ids_to_refetch(hdf5_filename, group_name)

    client = MongoClient(MONGO_URI)
    db = client["embedded-rift"]
    collection = db["games"]

    api = RiotAPI(api_key=RIOT_API_KEY)

    game_ids = split_game_ids(game_ids)
    refetch_games(game_ids[region], api, region, collection)


if __name__ == "__main__":
    main()
