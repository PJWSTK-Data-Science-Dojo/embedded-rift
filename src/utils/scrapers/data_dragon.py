from typing import Any

from utils.api_handler import APIHandler


PATCH = "15.4.1"
CHAMPIONS_DDRAGON_URL = (
    "https://ddragon.leagueoflegends.com/cdn/{}/data/en_US/champion.json"
)
CHAMPION_DDRAGON_URL = (
    "https://ddragon.leagueoflegends.com/cdn/{}/data/en_US/champion/{}.json"
)

ITEMS_DDRAGON_URL = "https://ddragon.leagueoflegends.com/cdn/{}/data/en_US/item.json"


class DataDragonChampionAPI(APIHandler):
    def __init__(self):
        super().__init__(rate_limit=60, rate_window=30)
        self.cache = {}

    def get_all_champions_data(
        self, patch: str = PATCH
    ) -> dict[str, str | dict[str, Any]]:
        cache_key = f"{patch}|champions"
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = CHAMPIONS_DDRAGON_URL.format(patch)
        data = self.get_json(url)
        self.cache[cache_key] = data
        return data

    def get_champion_data(self, champion_name: str, patch: str = PATCH) -> dict:
        data = self.get_all_champions_data(patch=patch)
        champion_id = None
        for champion in data["data"].values():
            if champion["name"].lower() == champion_name.lower():
                champion_id = champion["id"]
                break

        if champion_id is None:
            raise ValueError(f"Champion {champion_name} not found.")

        champion_name = champion_name.replace(" ", "").replace("'", "").replace(".", "")
        url = CHAMPION_DDRAGON_URL.format(patch, champion_id)

        cache_key = f"{patch}|{champion_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            data = self.get_json(url)
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {champion_id}")

        self.cache[cache_key] = data["data"][champion_id]
        return data["data"][champion_id]

    def all_champions_names(self, patch: str = PATCH) -> list[str]:
        champions = self.get_all_champions_data(patch=patch)

        names = [champion["name"] for champion in champions["data"].values()]
        return names


class DataDragonItemAPI(APIHandler):
    def __init__(self):
        super().__init__(rate_limit=60, rate_window=30)
        self.cache = {}

    def get_all_items(self, patch: str = PATCH) -> dict:
        cache_key = f"{patch}|items"
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = ITEMS_DDRAGON_URL.format(patch)
        data = self.get_json(url)
        self.cache[cache_key] = data["data"]
        return data["data"]

    def get_item_data(self, item_id: int | str, patch: str = PATCH) -> dict:
        if isinstance(item_id, int):
            item_id = str(item_id)

        items = self.get_all_items(patch=patch)

        if item_id not in items:
            raise ValueError(f"Item {item_id} not found.")

        return items[item_id]


class DataDragonAPI:
    def __init__(self):
        self.cache = {}
        self.champions = DataDragonChampionAPI()
        self.items = DataDragonItemAPI()

    @staticmethod
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
