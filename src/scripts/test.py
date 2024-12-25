from dotenv import load_dotenv
import os
import json
from datetime import datetime
from utils.riot_api import RiotAPI, Region


# Example Usage:
if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.getenv("RIOT_API")

    # Get PUUID by Riot ID (works across Riot Games titles)
    region = Region.EUROPE  # Region for Riot ID API (Europe-wide)
    riot_username = "Czarnuszka"  # Replace with the player's Riot ID username
    tag = "JDKZ"  # Replace with the player's tagline

    riot_api = RiotAPI(API_KEY)

    puuid = riot_api.get_puuid_by_riot_id(
        region=region, game_name=riot_username, tag=tag
    )

    print(f"PUUID from Riot ID: {puuid} for player {riot_username}#{tag}")

    # Get Match IDs by PUUID
    match_ids = riot_api.get_player_matches_ids(region=region, puuid=puuid)
    print(f"Match IDs: {match_ids}")

    last, *_ = match_ids

    match_result = riot_api.get_match_result(region=region, match_id=last)
    # print(f"Match Result: {match_result}")
    with open("match_result.json", "w") as f:
        json.dump(match_result, f, indent=4)

    # Get Match Timeline by Match ID
    match_timeline = riot_api.get_match_timeline(region=region, match_id=last)
    # print(f"Match Timeline: {match_timeline}")

    with open("match_timeline.json", "w") as f:
        json.dump(match_timeline, f, indent=4)

    match_data = riot_api.get_match_data(Region.EUROPE, last)

    with open(f"match_data.json", "w") as f:
        json.dump(match_data, f, indent=4)
