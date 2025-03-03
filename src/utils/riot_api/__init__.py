import requests
from datetime import datetime
from enum import StrEnum
import time

from utils.api_handler import APIHandler
from .match_parser.result import parse_match_result
from .match_parser.timeline import parse_timeline

QUEUE_ID = 420  # id of soloQ queue type (queue id's at https://static.developer.riotgames.com/docs/lol/queues.json)
QUEUE = "RANKED_SOLO_5x5"  # soloQ
GAME_TYPE = "ranked"


class Platform(StrEnum):
    EUW = "euw1"
    EUNE = "eun1"
    KR = "kr"
    NA = "na1"
    JAPAN = "jp1"
    BRAZIL = "br1"
    OCEANIA = "oc1"
    TURKEY = "tr1"
    RUSSIA = "ru"
    PHILIPPINES = "ph2"
    SINGAPORE = "sg2"
    THAILAND = "th2"
    TAIWAN = "tw2"
    VIETNAM = "vn2"
    LATIN1 = "la1"
    LATIN2 = "la2"

    def __str__(self):
        return self.value


class Region(StrEnum):
    AMERICA = "americas"
    EUROPE = "europe"
    ASIA = "asia"
    SEA = "sea"

    def __str__(self):
        return self.value


class Tier(StrEnum):
    IRON = "IRON"
    BRONZE = "BRONZE"
    SILVER = "SILVER"
    GOLD = "GOLD"
    PLATINUM = "PLATINUM"
    EMERALD = "EMERALD"
    DIAMOND = "DIAMOND"

    def __str__(self):
        return self.value


class Division(StrEnum):
    IV = "IV"
    III = "III"
    II = "II"
    I = "I"

    def __str__(self):
        return self.value


class RiotAPI(APIHandler):
    def __init__(self, api_key, rate_limit=100, rate_window=120):
        """
        :param self.api_key: Your Riot API key.
        :param rate_limit: Number of requests allowed in the rate window.
        :param rate_window: Time window in seconds for the rate limit.
        """
        super().__init__(rate_limit=rate_limit, rate_window=120)
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        self.request_times = []  # Tracks timestamps of API requests

    def get_json(self, url, *, headers=None, params=None):
        if headers is None:
            headers = {}

        headers = {**headers, "X-Riot-Token": self.api_key}
        return super().get_json(url, headers=headers, params=params)

    def get_puuid_by_riot_id(
        self, region: Region, game_name: str, tag: str
    ) -> str | None:
        url = f"https://{region.value}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag}"
        data = self.get_json(url)
        if not data:
            return None

        return data.get("puuid")

    def get_puuid_by_summoner_id(
        self, platform: Platform, summoner_id: str
    ) -> str | None:
        url = f"https://{platform.value}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"

        data = self.get_json(url)

        if not data:
            return None

        return data.get("puuid")

    def get_account_by_puuid(self, region: Region, puuid: str) -> dict:
        url = f"https://{region.value}.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"

        data = self.get_json(url)
        return data

    def get_player_matches_ids(
        self,
        region: Region,
        puuid: str,
        queue: int = QUEUE_ID,
        game_type: str = GAME_TYPE,
        count: int = None,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> list[str]:
        url = f"https://{region.value}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
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

        data = self.get_json(url, params=params)
        return data

    def get_apex_tiers_summoner_ids(self, platform: Platform) -> list[dict]:
        apex_leagues = ["challengerleagues", "grandmasterleagues", "masterleagues"]

        summoner_ids = []
        for league in apex_leagues:
            url = f"https://{platform.value}.api.riotgames.com/lol/league/v4/{league}/by-queue/{QUEUE}"
            data = self.get_json(url)
            if data:
                summoner_ids.extend(
                    [
                        {"summonerId": summoner["summonerId"]}
                        for summoner in data["entries"]
                    ]
                )
        return summoner_ids

    def get_summoners_page_by_rank(
        self, platform: Platform, tier: Tier, division: Division, page: int
    ) -> list[dict]:
        """
        :param platform:
        :param tier: One of ['DIAMOND', 'EMERALD', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON']
        :param division: One of ['I', 'II', 'III', 'IV']
        :param page: Page number to load
        :return: List of dictionaries
        """
        params = {
            "page": page,
        }

        url = f"https://{platform.value}.api.riotgames.com/lol/league/v4/entries/{QUEUE}/{tier.value}/{division.value}"
        data = self.get_json(url, params=params)

        return data

    def get_summoner_ids_by_rank(
        self, platform: Platform, tier: Tier, division: Division, count: int = -1
    ) -> list[dict]:
        """

        :param platform:
        :param tier: One of ['DIAMOND', 'EMERALD', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON']
        :param division: One of ['I', 'II', 'III', 'IV']
        :param count: How many pages to load. One page contains 205 summoner ids. Defaults to -1
        :return: List of dictionaries
        """
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
            data = self.get_json(url, params=params)

            if not data:
                break

            summoner_ids.extend(
                [{"summonerId": summoner["summonerId"]} for summoner in data]
            )

        return summoner_ids

    def get_summoners_count_by_rank(
        self, platform: Platform, tier: Tier, division: Division
    ) -> list[dict]:
        """
        :param platform:
        :param tier: One of ['DIAMOND', 'EMERALD', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON']
        :param division: One of ['I', 'II', 'III', 'IV']
        :return: List of dictionaries
        """
        url = f"https://{platform.value}.api.riotgames.com/lol/league/v4/entries/{QUEUE}/{tier.value}/{division.value}"

        low, high = 1, 1000  # Adjust `high` as needed for your system
        count = 0
        while low <= high:
            mid = (low + high) // 2

            params = {
                "page": mid,
            }

            data = self.get_json(url, params=params)
            count += 1

            if not data:
                high = mid - 1
            else:
                low = mid + 1

        print(
            f"Tier: {tier.value}, Division: {division.value}, Count: {(high - 1)*205 + len(data)}"
        )
        print(f"Total requests made: {count}")
        return (high - 1) * 205 + len(data)

    def get_match_result(self, region: Region, match_id) -> dict:
        url = (
            f"https://{region.value}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        )

        data = self.get_json(url)
        return data

    def get_match_timeline(self, region: Region, match_id: str) -> dict:
        url = f"https://{region.value}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"

        data = self.get_json(url)
        return data

    def get_match_data(self, region: Region, match_id: str) -> dict:
        # One dictionary containing important information from both
        # match/v5/matches/{match_id} and match/v5/matches/{match_id}/timeline endpoints
        match_result = self.get_match_result(region, match_id)
        timeline = self.get_match_timeline(region, match_id)

        parsed_match_result = parse_match_result(match_result)
        parsed_timeline = parse_timeline(timeline)

        match_data = {**parsed_match_result, "timeline": parsed_timeline}
        return match_data
