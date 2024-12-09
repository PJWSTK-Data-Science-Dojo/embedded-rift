from dotenv import load_dotenv
import os
import json
from . import RiotAPI, Division, Platform, Tier

load_dotenv()

API_KEY = os.getenv("RIOT_API")


def get_high_elo_summoner_ids(riot_api: RiotAPI, platform: Platform):
    apex_tiers_summoner_ids = riot_api.get_apex_tiers_summoner_ids(platform=platform)

    diamond_summoner_ids = []
    for division in Division:
        print(f"Division: {division.name}")
        summoner_ids = riot_api.get_summoner_ids_by_rank(
            platform=platform,
            tier=Tier.DIAMOND,
            division=division,
            count=-1,
        )
        diamond_summoner_ids.extend(summoner_ids)

    return [*apex_tiers_summoner_ids, *diamond_summoner_ids]


if __name__ == "__main__":
    riot_api = RiotAPI(API_KEY)
    summoner_ids = get_high_elo_summoner_ids(platform=Platform.EUW)

    with open("diamond+_summoner_ids.json", "w") as f:
        json.dump(summoner_ids, f, indent=4)
