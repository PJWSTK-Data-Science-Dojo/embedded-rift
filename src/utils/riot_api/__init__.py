import requests
from datetime import datetime
from enum import StrEnum
import time
from .match_result_parser import parse_match_result
from .timeline_parser import parse_timeline

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


class Region(StrEnum):
    AMERICA = "americas"
    EUROPE = "europe"
    ASIA = "asia"
    SEA = "sea"


class Tier(StrEnum):
    IRON = "IRON"
    BRONZE = "BRONZE"
    SILVER = "SILVER"
    GOLD = "GOLD"
    PLATINUM = "PLATINUM"
    EMERALD = "EMERALD"
    DIAMOND = "DIAMOND"


class Division(StrEnum):
    IV = "IV"
    III = "III"
    II = "II"
    I = "I"


class RiotAPI:
    def __init__(self, api_key, rate_limit=100, rate_window=120):
        """
        :param self.api_key: Your Riot API key.
        :param rate_limit: Number of requests allowed in the rate window.
        :param rate_window: Time window in seconds for the rate limit.
        """
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        self.request_times = []  # Tracks timestamps of API requests

    def _rate_limit_check(self):
        """
        Ensures the API requests stay within the specified rate limit.
        If the limit is exceeded, it pauses execution until a request can be made.
        """
        current_time = time.time()

        # Clean up old timestamps
        self.request_times = [
            t for t in self.request_times if current_time - t < self.rate_window
        ]

        if len(self.request_times) >= self.rate_limit:
            wait_time = self.rate_window - (current_time - self.request_times[0])
            # print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
            time.sleep(wait_time)

            self.request_times = [
                t for t in self.request_times if time.time() - t < self.rate_window
            ]

    def send_request(
        self, url: str, *, headers: dict = None, params: dict = None
    ) -> dict:
        """
        Sends a GET request to the Riot API while adhering to the rate limit.

        :param url: The endpoint URL.
        :param params: Optional query parameters for the request.
        :return: The response object from the request.
        """
        if headers is None:
            headers = {}

        self._rate_limit_check()  # Check and enforce rate limit

        headers = {**headers, "X-Riot-Token": self.api_key}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 429:
            self.request_times.append(time.time())
            return response.json()

        print(f"RateLimit Error: {response.status_code}, {response.text}")
        print(f"Waiting for {self.rate_window} seconds...")

        time.sleep(self.rate_window)

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 429:
            # TODO: Should we raise an exception here?
            raise Exception(
                f"Rate limit exceeded even after delay 121 seconds. {response.status_code}, {response.text}"
            )

        self.request_times.append(time.time())
        return response.json()

    def get_puuid_by_riot_id(
        self, region: Region, game_name: str, tag: str
    ) -> str | None:
        url = f"https://{region.value}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag}"
        data = self.send_request(url)
        if not data:
            return None

        return data.get("puuid")

    def get_puuid_by_summoner_id(
        self, platform: Platform, summoner_id: str
    ) -> str | None:
        url = f"https://{platform.value}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"

        data = self.send_request(url)

        if not data:
            return None

        return data.get("puuid")

    def get_account_by_puuid(self, region: Region, puuid: str) -> dict:
        url = f"https://{region.value}.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"

        data = self.send_request(url)
        return data

    def get_player_matches_ids(
        self,
        region: Region,
        puuid,
        queue=QUEUE_ID,
        game_type=GAME_TYPE,
        count: int = None,
        start_time: datetime = None,
        end_time: datetime = None,
    ):
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

        data = self.send_request(url, params=params)
        return data

    def get_apex_tiers_summoner_ids(self, platform: Platform) -> list[dict]:
        apex_leagues = ["challengerleagues", "grandmasterleagues", "masterleagues"]

        summoner_ids = []
        for league in apex_leagues:
            url = f"https://{platform.value}.api.riotgames.com/lol/league/v4/{league}/by-queue/{QUEUE}"
            data = self.send_request(url)
            if data:
                summoner_ids.extend(
                    [
                        {"summonerId": summoner["summonerId"]}
                        for summoner in data["entries"]
                    ]
                )
        return summoner_ids

    def get_summoner_ids_by_rank(
        self, platform: Platform, tier: Tier, division: Division, count: int = -1
    ) -> list[dict]:
        """

        :param self.api_key:
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
            data = self.send_request(url, params=params)

            if not data:
                break

            summoner_ids.extend(
                [{"summonerId": summoner["summonerId"]} for summoner in data]
            )

        return summoner_ids

    def get_match_result(self, region: Region, match_id) -> dict:
        url = (
            f"https://{region.value}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        )

        data = self.send_request(url)
        return data

    def get_match_timeline(self, region: Region, match_id: str) -> dict:
        url = f"https://{region.value}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"

        data = self.send_request(url)
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
