"""
HTML-based scraper for STS.pl League of Legends betting matches.
Fetches data by rendering the website and parsing the page content.
"""

import re
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup

from ._utils import BettingMatch, BettingMarket, BettingOdds

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class STSBettingHTMLScraper:
    """
    HTML scraper for STS.pl League of Legends betting.
    Fetches rendered page content using Playwright and parses betting data.
    """

    BASE_URL = "https://www.sts.pl/zaklady-bukmacherskie/esport/league-of-legends/lec/156/992"

    def get_html_content(self) -> Optional[str]:
        """
        Fetch the rendered HTML content of the LoL betting page using Playwright.

        Returns:
            HTML content as string, or None if fetch fails
        """
        if not sync_playwright:
            print("⚠ Playwright not installed. Install with: pip install playwright")
            return None

        try:
            with sync_playwright() as p:
                # Launch chromium browser in headless mode
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                # Navigate to page and wait for content to load
                page.goto(self.BASE_URL, wait_until="networkidle", timeout=30000)

                # Get the rendered HTML
                html_content = page.content()
                browser.close()
                return html_content

        except Exception as e:
            print(f"Failed to fetch HTML content: {e}")
            return None

    def parse_matches_from_html(self, html_content: str) -> list[BettingMatch]:
        """
        Parse League of Legends matches from HTML content.

        Args:
            html_content: Raw HTML from STS betting page

        Returns:
            List of BettingMatch objects
        """
        soup = BeautifulSoup(html_content, "lxml")
        page_text = soup.get_text()
        lines = [line.strip() for line in page_text.split("\n") if line.strip()]

        matches = []
        i = 0

        while i < len(lines):
            # Look for team separator "-"
            if lines[i] == "-" and i > 0 and i < len(lines) - 1:
                team1 = lines[i - 1]
                team2 = lines[i + 1]

                # Extract datetime and odds from next lines
                start_time = None
                team1_odds = None
                team2_odds = None

                for j in range(i + 2, min(i + 20, len(lines))):
                    line = lines[j]

                    # Parse datetime (dd.mm.yyyy, hh:mm format)
                    if re.match(r"\d{1,2}\.\d{1,2}\.\d{4}", line):
                        dt_match = re.match(
                            r"(\d{1,2})\.(\d{1,2})\.(\d{4}),\s*(\d{1,2}):(\d{2})", line
                        )
                        if dt_match:
                            day, month, year, hour, minute = dt_match.groups()
                            start_time = f"{year}-{month.zfill(2)}-{day.zfill(2)}T{hour.zfill(2)}:{minute.zfill(2)}:00"

                    # Parse odds (team1: "1" followed by odds value)
                    if line == "1" and j + 1 < len(lines):
                        try:
                            val = lines[j + 1].strip()
                            if re.match(r"^\d+\.\d{2}$", val):
                                team1_odds = float(val)
                        except (ValueError, IndexError):
                            pass

                    # Parse odds (team2: "2" followed by odds value)
                    if line == "2" and j + 1 < len(lines):
                        try:
                            val = lines[j + 1].strip()
                            if re.match(r"^\d+\.\d{2}$", val):
                                team2_odds = float(val)
                        except (ValueError, IndexError):
                            pass

                    # Stop looking when we hit next match or header
                    if (line == "-" or line.startswith("League of Legends")) and j > i + 5:
                        break

                # Create BettingMatch if we have all required data
                if start_time and team1_odds and team2_odds:
                    # Create odds objects
                    odds_list = [
                        BettingOdds(
                            id_oppty=None,
                            oppty_type=None,
                            outcome=team1,
                            odds=team1_odds,
                            is_active=True,
                            outcome_id=None,
                        ),
                        BettingOdds(
                            id_oppty=None,
                            oppty_type=None,
                            outcome=team2,
                            odds=team2_odds,
                            is_active=True,
                            outcome_id=None,
                        ),
                    ]

                    # Create market
                    market = BettingMarket(
                        market_type="match_winner",
                        market_type_id=None,
                        odds_list=odds_list,
                        description="Zwycięzca meczu (Match Winner)",
                    )

                    # Create and add match
                    match = BettingMatch(
                        match_id=len(matches) + 1,
                        team1=team1,
                        team2=team2,
                        tournament="League of Legends",
                        start_time=datetime.fromisoformat(start_time),
                        is_live=False,
                        markets={"match_winner": market},
                        match_url=self.BASE_URL,
                    )
                    matches.append(match)

            i += 1

        return matches

    def get_matches_from_html(self) -> list[BettingMatch]:
        """
        Fetch and parse League of Legends matches from STS website.

        Returns:
            List of BettingMatch objects
        """
        html_content = self.get_html_content()
        if not html_content:
            print("⚠ Failed to fetch HTML content from STS.pl")
            return []

        matches = self.parse_matches_from_html(html_content)
        if matches:
            print(f"✓ Scraped {len(matches)} League of Legends matches from STS.pl")
        else:
            print("⚠ No matches found on STS.pl")

        return matches
