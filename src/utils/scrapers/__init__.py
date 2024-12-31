from . import data_dragon as dd
from . import wiki
from . import _utils
import json


class LoLScraper:
    @staticmethod
    def get_all_champions() -> list[str]:
        return dd.all_champions()

    @staticmethod
    def get_champion_abilities(champion: str) -> list[str]:
        return wiki.get_champion_abilities(champion)

    @staticmethod
    def get_ability_data(champion: str, ability: str) -> str:
        return wiki.get_ability_data(champion, ability)

    @staticmethod
    def get_champion_data(champion: str) -> _utils.Champion:
        print(champion)
        champion_data = dd.get_champion_data(champion)
        champion_name = champion_data["name"]

        del champion_data["spells"]

        abi = wiki.get_champion_abilities(champion_name)
        abilities = {
            ability: _utils.ChampionAbility.from_dict(
                wiki.get_ability_data(champion_name, ability)
            )
            for ability in abi
        }
        champion_data["abilities"] = abilities

        return _utils.Champion(
            name=champion_name,
            stats=_utils.ChampionStats.from_dict(champion_data["stats"]),
            abilities=abilities,
            resource=dd.partype_to_resource(champion_data["partype"]),
            patch=dd.PATCH,
        )

    @staticmethod
    def get_all_champions_data() -> dict[str, _utils.Champion]:
        data = dd.get_all_champions_data()
        result = {}
        for champion in data["data"]:
            champion_data = LoLScraper.get_champion_data(champion)
            result[champion_data.name] = champion_data

        return result
