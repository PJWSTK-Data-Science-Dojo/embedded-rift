from typing import Any
import httpx
import json
from dataclasses import dataclass

from utils.api_handler import APIHandler


PATCH = "14.24.1"
DATA_DRAGON_URL = (
    "https://ddragon.leagueoflegends.com/cdn/14.24.1/data/en_US/champion.json"
)
CHAMPION_DRAGON_URL = (
    "https://ddragon.leagueoflegends.com/cdn/14.24.1/data/en_US/champion/{}.json"
)


class DataDragonAPI(APIHandler):
    def __init__(self):
        super().__init__(rate_limit=30, rate_window=30)

    def get_all_champions_data(self) -> dict[str, str | dict[str, Any]]:
        url = DATA_DRAGON_URL
        return self.get_json(url)

    def get_champion_data(self, champion_name: str) -> dict:
        champion_name = champion_name.replace(" ", "").replace("'", "").replace(".", "")
        url = CHAMPION_DRAGON_URL.format(champion_name)
        try:
            data = self.get_json(url)
        except:
            raise ValueError(f"Champion {champion_name} not found.")

        return data["data"][champion_name]

    def partype_to_resource(self, partype: str) -> str:
        resource = partype
        if resource in [
            "Rage",
            "Heat",
            "Grit",
            "Fury",
            "Shield",
            "Crimson Rush",
            "Courage",
            "Flow",
            "Frenzy",
            "Ferocity",
            "Blood Well",
        ]:
            return "MANALESS"

        return resource

    def all_champions(self) -> list[str]:
        champions = self.get_all_champions_data()
        return list(champions["data"].keys())
