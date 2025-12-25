"""
STS.pl betting scraper for League of Legends matches.
Uses HTML scraping to fetch betting data from the STS website.
"""

from ._utils import BettingMatch
from .sts_betting_html import STSBettingHTMLScraper


class STSBettingAPI:
    """
    STS.pl betting scraper for League of Legends matches.

    Uses HTML scraping to fetch League of Legends betting matches from STS.pl.
    This approach is reliable since the website renders content dynamically
    and the API endpoint returns empty data for LoL.
    """

    def __init__(self):
        """Initialize STS betting scraper."""
        self.html_scraper = STSBettingHTMLScraper()

    def get_matches_structured(self) -> list[BettingMatch]:
        """
        Get all League of Legends matches as structured dataclass objects.

        Returns:
            List of BettingMatch dataclass instances
        """
        return self.html_scraper.get_matches_from_html()

    def get_match_moneyline(self, match: BettingMatch):
        """
        Extract moneyline (match winner) odds from a match.

        Args:
            match: BettingMatch dataclass

        Returns:
            BettingMarket for match winner, or None if not available
        """
        # Try multiple possible moneyline market type names
        for market_key in [
            "match_winner",
            "match_winner_main",
            "match_winner_alt",
            "match_winner_special",
        ]:
            if market_key in match.markets:
                return match.markets[market_key]
        return None

    def get_match_handicaps(self, match: BettingMatch) -> list:
        """
        Extract all handicap/over-under markets from a match.

        Args:
            match: BettingMatch dataclass

        Returns:
            List of handicap/over-under BettingMarket objects
        """
        handicap_markets = []
        handicap_keywords = ["handicap", "over_under", "map_count", "spread"]

        for market_name, market in match.markets.items():
            if any(keyword in market_name for keyword in handicap_keywords):
                handicap_markets.append(market)

        return handicap_markets
