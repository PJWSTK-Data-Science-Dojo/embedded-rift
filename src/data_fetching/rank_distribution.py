from utils.riot_api import RiotAPI, Division, Tier, Region, Platform
from visualization.solo_q import plot_ccdf, plot_histogram
import json
from dotenv import load_dotenv
import os


def fetch_player_count_in_division_tier_for_platform(
    riot_api: RiotAPI, platform: Platform
) -> dict:
    data = {}
    for tier in Tier:
        for division in Division:
            print(f"Getting {tier} {division}")
            response = riot_api.get_summoner_ids_by_rank(
                platform=platform, tier=tier, division=division
            )
            data[f"{tier}_{division}"] = len(response)

    return data


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("RIOT_API")

    riot_api = RiotAPI(API_KEY)

    for platform in Platform:
        print(f"Getting {platform}")
        data = fetch_player_count_in_division_tier_for_platform(
            riot_api=riot_api, platform=platform
        )
        with open(f"data/{platform}_data.json", "w") as f:
            json.dump(data, f, indent=4)

        plot_ccdf(data, f"ccdf_{platform}.png")
        plot_histogram(
            data, f"histo_{platform}.png", title=f"Histogram of players in {platform}"
        )
