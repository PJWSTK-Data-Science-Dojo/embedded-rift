from utils.riot_api import RiotAPI, Division, Tier, Region, Platform
from visualization.solo_q import plot_ccdf, plot_histogram
import json
from dotenv import load_dotenv
import os


def get_players_in_rank(
    riot_api: RiotAPI, platform: Platform
) -> dict:
    data = {}
    for tier in Tier:
        for division in Division:
            print(f"Getting {tier} {division}")
            count = riot_api.get_summoners_count_by_rank(
                platform=platform, tier=tier, division=division
            )
            data[f"{tier}_{division}"] = count

    return data


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("RIOT_API")

    riot_api = RiotAPI(API_KEY)
    platform = Platform.EUW
    
    print(f"Getting {platform}")
    data = get_players_in_rank(
        riot_api=riot_api, platform=platform
    )
    with open(f"data/{platform}_data.json", "w") as f:
        json.dump(data, f, indent=4)

    plot_ccdf(data, f"viz/ccdf_{platform}.png")
    plot_histogram(
        data, f"viz/histo_{platform}.png", title=f"Histogram of players in {platform}"
    )
