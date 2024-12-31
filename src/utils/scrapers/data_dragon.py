import httpx
import json
from dataclasses import dataclass


PATCH = "14.24.1"
DATA_DRAGON_URL = (
    "https://ddragon.leagueoflegends.com/cdn/14.24.1/data/en_US/champion.json"
)
CHAMPION_DRAGON_URL = (
    "https://ddragon.leagueoflegends.com/cdn/14.24.1/data/en_US/champion/{}.json"
)


def get_all_champions_data() -> dict:
    url = DATA_DRAGON_URL
    response = httpx.get(url)
    return response.json()


def get_champion_data(champion_name: str) -> dict:
    champion_name = champion_name.replace(" ", "").replace("'", "").replace(".", "")
    url = CHAMPION_DRAGON_URL.format(champion_name)
    response = httpx.get(url)
    if response.status_code != 200:
        print(response.text)
        raise ValueError(f"Champion {champion_name} not found.")
    data = response.json()
    return data["data"][champion_name]


def partype_to_resource(partype: str) -> str:
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


def all_champions():
    champions = get_all_champions_data()
    return champions["data"].keys()
