"""
Superbet.pl betting scraper for League of Legends matches.
Uses the Superbet API to fetch betting data.
"""

import requests
from datetime import datetime, timedelta
from typing import Optional
from ._utils import BettingMatch, BettingMarket, BettingOdds


# Superbet Market Types Mapping
SUPERBET_MARKET_TYPES = {
    2519: "match_winner",  # Zwycięzca (Winner)
}


class SuperbetBettingAPI:
    """
    Superbet.pl betting scraper for League of Legends matches.

    Uses the Superbet API to fetch League of Legends betting matches.
    This approach is reliable since Superbet provides a structured JSON API.
    """

    BASE_URL = "https://production-superbet-offer-pl.freetls.fastly.net/v2/pl-PL"
    SPORT_ID = 39  # League of Legends
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://superbet.pl",
    }

    def __init__(self):
        """Initialize Superbet betting scraper."""
        self.session = requests.Session()

    def get_matches_structured(self) -> list[BettingMatch]:
        """
        Get all League of Legends matches as structured dataclass objects.

        Returns:
            List of BettingMatch dataclass instances
        """
        # Fetch data for a wide date range (today to 1 year from now)
        now = datetime.utcnow()
        start_date = now
        end_date = now + timedelta(days=365)

        url = f"{self.BASE_URL}/events/by-date"
        params = {
            "offerState": "prematch",
            "startDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "endDate": end_date.strftime("%Y-%m-%d %H:%M:%S"),
            "sportId": self.SPORT_ID,
        }

        try:
            response = self.session.get(url, params=params, headers=self.HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("error") and "data" in data:
                return self._parse_matches(data["data"])
            return []
        except requests.RequestException as e:
            print(f"Error fetching Superbet data: {e}")
            return []

    def _parse_matches(self, raw_matches: list) -> list[BettingMatch]:
        """
        Parse raw API matches into BettingMatch dataclass objects.

        Args:
            raw_matches: List of match dictionaries from API response

        Returns:
            List of parsed BettingMatch objects
        """
        matches = []

        for match_data in raw_matches:
            try:
                match_id = match_data.get("eventId")
                match_name = match_data.get("matchName", "")
                match_date = match_data.get("matchDate")
                tournament_id = match_data.get("tournamentId")
                odds_list = match_data.get("odds", [])

                # Parse team names from matchName (e.g., "Los Ratones·Fnatic")
                teams = match_name.split("·")
                team1 = teams[0].strip() if len(teams) > 0 else "Unknown"
                team2 = teams[1].strip() if len(teams) > 1 else "Unknown"

                # Parse match date
                try:
                    start_time = datetime.strptime(match_date, "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    start_time = datetime.utcnow()

                # Check if match is live
                is_live = match_data.get("metadata", {}).get("status") == "STARTED"

                # Parse odds/markets
                markets = self._parse_markets(odds_list)

                # Create BettingMatch object
                betting_match = BettingMatch(
                    match_id=match_id,
                    team1=team1,
                    team2=team2,
                    tournament=f"Tournament {tournament_id}",
                    start_time=start_time,
                    is_live=is_live,
                    markets=markets,
                    match_url=None,
                )

                matches.append(betting_match)

            except Exception as e:
                print(f"Error parsing match {match_data.get('matchName', 'Unknown')}: {e}")
                continue

        return matches

    def _parse_markets(self, odds_list: list) -> dict[str, BettingMarket]:
        """
        Parse odds array into BettingMarket objects.

        Args:
            odds_list: List of odds from API response

        Returns:
            Dictionary mapping market_type to BettingMarket objects
        """
        markets = {}

        # Group odds by market (they should all be Zwycięzca/match_winner)
        market_groups = {}

        for odd_data in odds_list:
            market_id = odd_data.get("marketId", 2519)
            market_type = SUPERBET_MARKET_TYPES.get(market_id, f"market_{market_id}")

            if market_type not in market_groups:
                market_groups[market_type] = []

            # Create BettingOdds object
            betting_odd = BettingOdds(
                id_oppty=odd_data.get("outcomeId", 0),
                oppty_type=market_id,
                outcome=odd_data.get("name", ""),
                odds=float(odd_data.get("price", 1.0)),
                is_active=odd_data.get("status") == "active",
                outcome_id=odd_data.get("outcomeId"),
            )

            market_groups[market_type].append(betting_odd)

        # Create BettingMarket objects from grouped odds
        for market_type, odds_list_grouped in market_groups.items():
            market_id = 2519  # Superbet uses marketId 2519 for match winner
            markets[market_type] = BettingMarket(
                market_type=market_type,
                market_type_id=market_id,
                odds_list=odds_list_grouped,
                description="Match Winner (Zwycięzca)",
            )

        return markets

    def get_match_moneyline(self, match: BettingMatch) -> Optional[BettingMarket]:
        """
        Extract moneyline (match winner) odds from a match.

        Args:
            match: BettingMatch dataclass

        Returns:
            BettingMarket for match winner, or None if not available
        """
        if "match_winner" in match.markets:
            return match.markets["match_winner"]
        return None
