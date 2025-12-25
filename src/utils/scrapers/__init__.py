from . import data_dragon
from .wiki import WikiScraper
from .sts_betting import STSBettingAPI
from .superbet_betting import SuperbetBettingAPI
from .betclic_betting import BetclicBettingAPI
from . import _utils


class LoLScraper:
    def __init__(self):
        self.ddragon = data_dragon.DataDragonAPI()
        self.wiki = WikiScraper()
        self.betting = STSBettingAPI()
        self.betting_superbet = SuperbetBettingAPI()
        self.betting_betclic = BetclicBettingAPI()

    def get_champions_names(self, patch: str = data_dragon.PATCH) -> list[str]:
        return self.ddragon.champions.all_champions_names(patch)

    def get_champion_data(
        self, champion_name: str, patch: str = data_dragon.PATCH
    ) -> dict:
        return self.ddragon.champions.get_champion_data(champion_name, patch)

    def get_items(self, patch: str = data_dragon.PATCH) -> dict:
        return self.ddragon.items.get_all_items(patch)

    def get_item_data(self, item_id: int | str, patch: str = data_dragon.PATCH) -> dict:
        return self.ddragon.items.get_item_data(item_id, patch)

    def get_champion_abilities(self, champion: str) -> list[str]:
        return self.wiki.get_champion_ability_names(champion)

    def get_ability_data(self, champion: str, ability: str) -> str:
        return self.wiki.get_ability_data(champion, ability)

    def get_champion_data(self, champion: str) -> _utils.Champion:
        champion_data = self.ddragon.champions.get_champion_data(champion)
        champion_name = champion_data["name"]

        del champion_data["spells"]

        abi = self.wiki.get_champion_ability_names(champion_name)
        abilities = {
            ability: _utils.ChampionAbility.from_dict(
                self.wiki.get_ability_data(champion_name, ability)
            )
            for ability in abi
        }
        champion_data["abilities"] = abilities

        return _utils.Champion(
            name=champion_name,
            stats=_utils.ChampionStats.from_dict(champion_data["stats"]),
            abilities=abilities,
            resource=self.ddragon.partype_to_resource(champion_data["partype"]),
            patch=data_dragon.PATCH,
        )

    def get_all_champions_data(self) -> dict[str, _utils.Champion]:
        data = self.ddragon.champions.get_all_champions_data()
        result = {}
        for champion in data["data"]:
            champion_data: _utils.Champion = self.get_champion_data(champion)
            result[champion_data.name] = champion_data

        return result

    # Betting-related methods
    def get_betting_matches(self) -> list[_utils.BettingMatch]:
        """
        Get all available League of Legends betting matches from STS.pl.

        Returns:
            List of BettingMatch dataclass instances with betting odds
        """
        return self.betting.get_matches_structured()

    def get_match_odds(
        self, match: _utils.BettingMatch, market_type: str
    ) -> _utils.BettingMarket:
        """
        Get odds for a specific market type from a match.

        Args:
            match: BettingMatch dataclass
            market_type: Market type name (e.g., "match_winner")

        Returns:
            BettingMarket for the specified type, or None if not available
        """
        return match.markets.get(market_type)

    def get_betting_matches_superbet(self) -> list[_utils.BettingMatch]:
        """
        Get all available League of Legends betting matches from Superbet.pl.

        Returns:
            List of BettingMatch dataclass instances with betting odds
        """
        return self.betting_superbet.get_matches_structured()

    def get_match_odds_superbet(self, match: _utils.BettingMatch) -> _utils.BettingMarket:
        """
        Get moneyline odds for a match from Superbet.

        Args:
            match: BettingMatch dataclass

        Returns:
            BettingMarket for match winner, or None if not available
        """
        return self.betting_superbet.get_match_moneyline(match)

    def get_betting_matches_betclic(self) -> list[_utils.BettingMatch]:
        """
        Get all available League of Legends betting matches from Betclic.pl.

        Returns:
            List of BettingMatch dataclass instances with betting odds
        """
        return self.betting_betclic.get_matches_structured()

    def get_match_odds_betclic(self, match: _utils.BettingMatch) -> _utils.BettingMarket:
        """
        Get moneyline odds for a match from Betclic.

        Args:
            match: BettingMatch dataclass

        Returns:
            BettingMarket for match winner, or None if not available
        """
        return self.betting_betclic.get_match_moneyline(match)
