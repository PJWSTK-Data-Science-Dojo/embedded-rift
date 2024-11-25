from dotenv import load_dotenv
import os
from PoroPilot import PoroPilot
import requests


def get_puuid_by_riot_id(api_key, region, game_name, tag):
    url = f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag}"
    headers = {"X-Riot-Token": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data.get("puuid")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def get_player_matches_ids(api_key, region, puuid):
    url = (
        f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    )
    headers = {"X-Riot-Token": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def get_match_timeline(api_key, region, match_id):
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"

    headers = {"X-Riot-Token": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
