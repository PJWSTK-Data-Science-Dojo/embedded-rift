from dotenv import load_dotenv
import os
import json
from datetime import datetime
from utils.riot_api import *

load_dotenv()

API_KEY = os.getenv("RIOT_API")

# Example Usage:
if __name__ == "__main__":
    # Get PUUID by Riot ID (works across Riot Games titles)
    region = Region.EUROPE  # Region for Riot ID API (Europe-wide)
    riot_username = "Czarnuszka"  # Replace with the player's Riot ID username
    tag = "JDKZ"  # Replace with the player's tagline
    puuid = get_puuid_by_riot_id(
        api_key=API_KEY, region=region, game_name=riot_username, tag=tag
    )
    # print(f"PUUID from Riot ID: {puuid}")

    # Get Match IDs by PUUID
    match_ids = get_player_matches_ids(api_key=API_KEY, region=region, puuid=puuid)
    # print(f"Match IDs: {match_ids}")

    last, *_ = match_ids

    match_result = get_match_result(api_key=API_KEY, region=region, match_id=last)
    # print(f"Match Result: {match_result}")
    with open("match_result.json", "w") as f:
        json.dump(match_result, f, indent=4)

    # Get Match Timeline by Match ID
    match_timeline = get_match_timeline(api_key=API_KEY, region=region, match_id=last)
    # print(f"Match Timeline: {match_timeline}")

    with open("match_timeline.json", "w") as f:
        json.dump(match_timeline, f, indent=4)

    match_data = get_match_data(API_KEY, Region.EUROPE, last)
    with open(f"match_data.json", 'w') as f:
        json.dump(match_data, f, indent=4)
