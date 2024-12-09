from dotenv import load_dotenv
import os
from PoroPilot import PoroPilot
import requests
from datetime import datetime
from enum import Enum
import time
from src.utils.match_result_parser import parse_match_result
from src.utils.timeline_parser import parse_timeline

QUEUE_ID = 420  # id of soloQ queue type (queue id's at https://static.developer.riotgames.com/docs/lol/queues.json)
QUEUE = 'RANKED_SOLO_5x5'  # soloQ
GAME_TYPE = "ranked"


class Platform(Enum):
    EUW = 'euw1'
    EUNE = 'eun1'
    KR = 'kr'
    NA = 'na1'
    JAPAN = 'jp1'
    BRAZIL = 'br1'
    OCEANIA = 'oc1'
    TURKEY = 'tr1'
    RUSSIA = 'ru'
    PHILIPPINES = 'ph2'
    SINGAPORE = 'sg2'
    THAILAND = 'th2'
    TAIWAN = 'tw2'
    VIETNAM = 'vn2'
    LATIN1 = 'la1'
    LATIN2 = 'la2'


class Region(Enum):
    AMERICA = 'americas'
    EUROPE = 'europe'
    ASIA = 'asia'
    SEA = 'sea'


class Tier(Enum):
    IRON = 'IRON'
    BRONZE = 'BRONZE'
    SILVER = 'SILVER'
    GOLD = 'GOLD'
    PLATINUM = 'PLATINUM'
    EMERALD = 'EMERALD'
    DIAMOND = 'DIAMOND'


class Division(Enum):
    I = 'I'
    II = 'II'
    III = 'III'
    IV = 'IV'


def send_request(endpoint, headers, params=None, depth=0):
    response = requests.get(endpoint, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429 and depth < 5:
        # Current api rate is 100 requests every 2 minutes
        # When 'Rate limit exceeded' is returned, send another request after 120 seconds
        delay = 121
        print(f"Error: {response.status_code}, {response.text}")
        print(f"Waiting for {delay} seconds...")
        time.sleep(delay)
        # Repeated requests are send recursively up to some max depth (currently 5)
        return send_request(endpoint, headers, params, depth+1)
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def get_puuid_by_riot_id(api_key, region: Region, game_name, tag):
    url = f"https://{region.value}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag}"
    headers = {"X-Riot-Token": api_key}

    data = send_request(url, headers)
    if data:
        return data.get("puuid")
    else:
        return data


def get_puuid_by_summoner_id(api_key, platform: Platform, summoner_id):
    url = f"https://{platform.value}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
    headers = {"X-Riot-Token": api_key}

    data = send_request(url, headers)
    if data:
        return data.get("puuid")
    else:
        return data


def get_account_by_puuid(api_key, region: Region, puuid):
    url = f"https://{region.value}.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"
    headers = {"X-Riot-Token": api_key}

    data = send_request(url, headers)
    return data


def get_player_matches_ids(api_key, region: Region, puuid, queue=QUEUE_ID, game_type=GAME_TYPE, count: int = None,
                           start_time: datetime = None, end_time: datetime = None):
    url = f"https://{region.value}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
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

    data = send_request(url, headers, params)
    return data


def get_apex_tiers_summoner_ids(api_key, platform: Platform):
    headers = {"X-Riot-Token": api_key}
    apex_leagues = ['challengerleagues', 'grandmasterleagues', 'masterleagues']

    summoner_ids = []
    for league in apex_leagues:
        url = f"https://{platform.value}.api.riotgames.com/lol/league/v4/{league}/by-queue/{QUEUE}"
        data = send_request(url, headers)
        if data:
            summoner_ids.extend([{'summonerId': summoner['summonerId']} for summoner in data['entries']])
    return summoner_ids


def get_summoner_ids_by_rank(api_key, platform: Platform, tier: Tier, division: Division, count=-1):
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

        url = f"https://{platform.value}.api.riotgames.com/lol/league/v4/entries/{QUEUE}/{tier.value}/{division.value}"
        data = send_request(url, headers, params)
        if data:
            summoner_ids.extend([{'summonerId': summoner['summonerId']} for summoner in data])
        else:
            break
    return summoner_ids


def get_match_result(api_key, region: Region, match_id):
    url = f"https://{region.value}.api.riotgames.com/lol/match/v5/matches/{match_id}"

    headers = {"X-Riot-Token": api_key}
    data = send_request(url, headers)
    return data


def get_match_timeline(api_key, region: Region, match_id):
    url = f"https://{region.value}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"

    headers = {"X-Riot-Token": api_key}

    data = send_request(url, headers)
    return data


def get_match_data(api_key, region: Region, match_id):
    # One dictionary containing important information from both
    # match/v5/matches/{match_id} and match/v5/matches/{match_id}/timeline endpoints
    match_result = get_match_result(api_key, region, match_id)
    timeline = get_match_timeline(api_key, region, match_id)

    parsed_match_result = parse_match_result(match_result)
    parsed_timeline = parse_timeline(timeline)

    match_data = {**parsed_match_result, "timeline": parsed_timeline}
    return match_data
