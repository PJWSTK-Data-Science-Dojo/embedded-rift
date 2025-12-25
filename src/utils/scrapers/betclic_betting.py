"""
Betclic.pl betting scraper for League of Legends matches.
Uses Camoufox (stealthy Firefox) and XPath-based parsing.
"""

import re
import json
from datetime import datetime, timedelta
from typing import Optional
from lxml import html
from ._utils import BettingMatch, BettingMarket, BettingOdds

try:
    from camoufox.sync_api import Camoufox
except ImportError:
    Camoufox = None


class BetclicBettingAPI:
    """
    Betclic.pl betting scraper for League of Legends matches.

    Uses Camoufox to fetch and lxml to parse matches via XPaths.
    """

    BASE_URL = "https://www.betclic.pl/league-of-legends-slol"

    def get_html_content(self) -> Optional[str]:
        """
        Fetch the rendered HTML content using Camoufox.

        Returns:
            HTML content as string, or None if fetch fails
        """
        if not Camoufox:
            print("⚠ Camoufox not installed. Install with: uv pip install camoufox")
            return None

        try:
            with Camoufox(headless=True, humanize=True) as browser:
                page = browser.new_page()
                print(f"  [Betclic] Navigating to {self.BASE_URL} using Camoufox...")
                page.goto(self.BASE_URL, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(5000)
                
                title = page.title()
                html_content = page.content()
                
                if "Forbidden" in title or "Error" in title or "Access Denied" in title:
                    print(f"  [Betclic] Access Denied. Title: {title}")
                
                return html_content

        except Exception as e:
            print(f"Failed to fetch HTML content with Camoufox: {e}")
            return None

    def get_matches_structured(self) -> list[BettingMatch]:
        """
        Get all League of Legends matches as structured dataclass objects.

        Returns:
            List of BettingMatch dataclass instances
        """
        html_str = self.get_html_content()
        if not html_str:
            return []
        return self.parse_matches_from_html(html_str)

    def parse_matches_from_html(self, html_content: str) -> list[BettingMatch]:
        """
        Parse League of Legends matches from HTML content using XPaths.

        Args:
            html_content: Raw HTML from Betclic betting page

        Returns:
            List of BettingMatch objects
        """
        try:
            tree = html.fromstring(html_content)
        except Exception as e:
            print(f"Error parsing HTML with lxml: {e}")
            return []

        matches = []
        match_id = 1000

        # Match containers
        match_elements = tree.xpath("//a[contains(@class, 'cardEvent')]")
        
        for el in match_elements:
            try:
                # XPaths relative to match container
                team1_list = el.xpath(".//scoreboards-scoreboard-global/div[1]/div/text()")
                team2_list = el.xpath(".//scoreboards-scoreboard-global/div[3]/div/text()")
                time_list = el.xpath(".//scoreboards-scoreboard-global/div[2]/div/text()")
                tournament_list = el.xpath(".//bcdk-breadcrumb-item[last()]//span/text()")
                
                # Odds buttons
                odd1_list = el.xpath(".//button[contains(@class, 'is-odd')][1]//span[@class='btn_label' and not(contains(@class, 'is-top'))]/text()")
                odd2_list = el.xpath(".//button[contains(@class, 'is-odd')][2]//span[@class='btn_label' and not(contains(@class, 'is-top'))]/text()")

                if not (team1_list and team2_list and time_list):
                    continue

                team1 = team1_list[0].strip()
                team2 = team2_list[0].strip()
                time_str = time_list[0].strip()
                tournament = tournament_list[0].strip() if tournament_list else "Unknown"
                
                odd1_str = odd1_list[0].strip() if odd1_list else "1.00"
                odd2_str = odd2_list[0].strip() if odd2_list else "1.00"

                # Parse time (HH:MM)
                try:
                    hour, minute = map(int, time_str.split(':'))
                    time_obj = datetime.strptime(f"{hour}:{minute}", "%H:%M").time()
                    now = datetime.now()
                    start_datetime = datetime.combine(now.date(), time_obj)
                    if start_datetime < now:
                        start_datetime += timedelta(days=1)
                except ValueError:
                    start_datetime = datetime.now()

                # Parse odds
                odd1 = float(odd1_str.replace(",", "."))
                odd2 = float(odd2_str.replace(",", "."))

                market_odds = [
                    BettingOdds(outcome=team1, odds=odd1, is_active=True),
                    BettingOdds(outcome=team2, odds=odd2, is_active=True)
                ]

                betting_match = BettingMatch(
                    match_id=match_id,
                    team1=team1,
                    team2=team2,
                    tournament=tournament,
                    start_time=start_datetime,
                    is_live=False,
                    markets={
                        "match_winner": BettingMarket(
                            market_type="match_winner",
                            market_type_id=1,
                            odds_list=market_odds,
                            description="Match Winner"
                        )
                    }
                )

                matches.append(betting_match)
                match_id += 1

            except Exception as e:
                print(f"Error parsing individual match: {e}")
                continue

        return matches

    def get_match_moneyline(self, match: BettingMatch) -> Optional[BettingMarket]:
        """Extract moneyline odds."""
        return match.markets.get("match_winner")
