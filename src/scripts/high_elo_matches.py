import random
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from utils.riot_api import RiotAPI, Platform, Region


def save_point_generator(save_points):
    """
    Generator that yields the next save point based on the current progress.

    :param save_points: List of save points to track progress.
    :yield: Next save point.
    """
    for point in save_points:
        yield point


def get_high_elo_matches(
    api: RiotAPI,
    platform: Platform,
    region: Region,
    save_points=[1000, 5000, 10000, 15000],
):
    """
    Fetches up to max_count Challenger games using the RiotAPI class and saves progress to JSON files.

    :param api: Instance of RiotAPI.
    :param platform: Platform enum value.
    :param region: Region enum value.
    :param max_count: Maximum number of matches to fetch.
    :return: None.
    """
    print("Fetching HighElo players...")
    high_elo_players = api.get_apex_tiers_summoner_ids(platform)

    if not high_elo_players:
        print("No HighElo players found.")
        return

    print(f"Found {len(high_elo_players)} HighElo players.")

    with open("data/high_elo_players.json", "w") as f:
        json.dump(high_elo_players, f, indent=4)

    # Shuffle the players to ensure randomness
    random.shuffle(high_elo_players)

    match_ids = set()
    start_time = datetime.now() - timedelta(days=7)

    save_point_gen = save_point_generator(save_points)
    next_save_point = next(save_point_gen, None)

    for player in high_elo_players:
        summoner_id = player["summonerId"]

        print(f"Fetching PUUID for summoner ID: {summoner_id}")
        puuid = api.get_puuid_by_summoner_id(platform, summoner_id)

        if not puuid:
            print(f"Failed to get PUUID for summoner ID: {summoner_id}")
            continue

        print(f"Fetching matches for PUUID: {puuid}")
        matches = api.get_player_matches_ids(
            region, puuid, start_time=start_time, count=10
        )

        if matches:
            match_ids.update(matches)

        print(f"Collected {len(match_ids)} unique matches so far.")

        # Save progress when reaching the next save point
        if next_save_point and len(match_ids) >= next_save_point:
            file_name = f"data/high_elo_matches_{next_save_point}.json"
            with open(file_name, "w") as f:
                json.dump(list(match_ids), f, indent=4)
            print(f"Saved {len(match_ids)} matches to {file_name}.")
            next_save_point = next(save_point_gen, None)

        # Stop if max_count is reached
        if next_save_point is None:
            break

    # Final save if not already saved
    file_name = f"data/high_elo_matches_final_{len(match_ids)}.json"
    with open(file_name, "w") as f:
        json.dump(list(match_ids), f, indent=4)
    print(f"Saved final {len(match_ids)} matches to {file_name}.")
    return list(match_ids)


def fetch_and_save_match_data(
    api: RiotAPI,
    region: Region,
    match_ids: list,
    save_points=[1000, 5000, 10000, 15000],
):
    """
    Fetch detailed match data for a list of match IDs and save progress to JSON files in line-delimited format.

    :param api: Instance of RiotAPI.
    :param region: Region enum value.
    :param match_ids: List of match IDs to fetch data for.
    :param max_count: Maximum number of matches to process.
    :return: None.
    """
    save_point_gen = save_point_generator(save_points)
    next_save_point = next(save_point_gen, None)

    processed_count = 0
    match_data = []

    for match_id in match_ids:
        print(f"Fetching data for match ID: {match_id}")
        match_data.append(api.get_match_result(region, match_id))
        processed_count += 1

        print(f"Processed {processed_count} matches so far.")

        # Save progress at save points
        if next_save_point and processed_count >= next_save_point:
            file_name = f"data/high_elo_match_data_{next_save_point}.jsonl"
            with open(file_name, "w") as f:
                for match in match_data:
                    if match:
                        f.write(json.dumps(match) + "\n")

            print(f"Saved {processed_count} matches to {file_name}.")
            next_save_point = next(save_point_gen, None)

        # Stop if max_count is reached
        if next_save_point is None:
            break

    print(f"Finished processing {processed_count} matches.")
    # Final save if not already saved
    if not match_data:
        return

    file_name = f"data/high_elo_match_data_final_{processed_count}.jsonl"
    with open(file_name, "w") as f:
        for match in match_data:
            if match:
                f.write(json.dumps(match) + "\n")
    print(f"Saved final {processed_count} matches to {file_name}.")


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("RIOT_API")

    riot_api = RiotAPI(api_key=API_KEY)
    # save_points = [10, 50, 100]
    save_points = [10, 50, 100, 500, 1000, 2500, 5000, 10000, 20000]
    # Define platforms and regions for Challenger queue
    PLATFORM = Platform.EUW  # Change to your preferred platform
    REGION = Region.EUROPE  # Corresponding region for the platform

    random_matches = get_high_elo_matches(
        riot_api, PLATFORM, REGION, save_points=save_points
    )
    fetch_and_save_match_data(riot_api, REGION, random_matches, save_points=save_points)

    # Example: Save or process the data
    # for match in random_matches:
    #     print(match)
