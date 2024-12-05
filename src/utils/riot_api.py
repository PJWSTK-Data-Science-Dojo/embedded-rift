from dotenv import load_dotenv
import os
from PoroPilot import PoroPilot
import requests
from datetime import datetime

QUEUE_ID = 420  # id of soloQ queue type (queue id's at https://static.developer.riotgames.com/docs/lol/queues.json)
QUEUE = 'RANKED_SOLO_5x5'  # soloQ
GAME_TYPE = "ranked"


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


def get_puuid_by_summoner_id(api_key, platform, summoner_id):
    url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
    headers = {"X-Riot-Token": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data.get("puuid")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def get_account_by_puuid(api_key, region, puuid):
    url = f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"
    headers = {"X-Riot-Token": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def get_player_matches_ids(api_key, region, puuid, queue=QUEUE_ID, game_type=GAME_TYPE, count: int = None,
                           start_time: datetime = None, end_time: datetime = None):
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    headers = {"X-Riot-Token": api_key}
    params = {
        "queue": queue,
        "type": game_type,
    }
    if count:
        params["count"] = count
    if start_time:
        params["startTime"] = round(start_time.timestamp())
    if end_time:
        params["endTime"] = round(end_time.timestamp())

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def get_apex_tiers_summoner_ids(api_key, platform):
    headers = {"X-Riot-Token": api_key}
    apex_leagues = ['challengerleagues', 'grandmasterleagues', 'masterleagues']

    summoner_ids = []
    for league in apex_leagues:
        url = f"https://{platform}.api.riotgames.com/lol/league/v4/{league}/by-queue/{QUEUE}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()['entries']
            summoner_ids.extend([{'summonerId': summoner['summonerId']} for summoner in data])
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break
    return summoner_ids


def get_summoner_ids_by_rank(api_key, platform, tier, division, count=-1):
    """

    :param api_key:
    :param platform:
    :param tier: One of ['DIAMOND', 'EMERALD', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON']
    :param division: One of ['I', 'II', 'III', 'IV']
    :param count: How many pages to load. One page contains 205 summoner ids. Defaults to -1
    :return: List of dictionaries
    """
    headers = {"X-Riot-Token": api_key}
    page = 1

    limit = True
    if count < 0:
        limit = False

    summoner_ids = []
    while not limit or page <= count:
        params = {
            "page": page,
        }
        page += 1

        url = f"https://{platform}.api.riotgames.com/lol/league/v4/entries/{QUEUE}/{tier}/{division}"

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                summoner_ids.extend([{'summonerId': summoner['summonerId']} for summoner in data])
            else:
                break
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break
    return summoner_ids


def get_match_result(api_key, region, match_id):
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"

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
