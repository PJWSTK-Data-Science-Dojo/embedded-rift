from dotenv import load_dotenv
import os
import json
from riot_api import *

load_dotenv()

API_KEY = os.getenv("RIOT_API")


# retrieves summoner_id of every diamond+ player on given platform (about 60k ids on EUW)
def get_high_elo_summoner_ids(api_key, platform: Platform):
    apex_tiers_summoner_ids = get_apex_tiers_summoner_ids(api_key=api_key, platform=platform)

    diamond_summoner_ids = []
    for division in Division:
        print(f'Division: {division.name}')
        summoner_ids = get_summoner_ids_by_rank(api_key=api_key, platform=platform, tier=Tier.DIAMOND, division=division, count=-1)
        diamond_summoner_ids.extend(summoner_ids)

    return [*apex_tiers_summoner_ids, *diamond_summoner_ids]


summoner_ids = get_high_elo_summoner_ids(api_key=API_KEY, platform=Platform.EUW)

with open("diamond+_summoner_ids.json", "w") as f:
    json.dump(summoner_ids, f, indent=4)
