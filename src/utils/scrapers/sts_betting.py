"""
STS.pl betting scraper for League of Legends matches.
Uses HTML scraping to fetch betting data from the STS website.
"""

import re
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup

from ._utils import BettingMatch, BettingMarket, BettingOdds

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class STSBettingAPI:
    """
    STS.pl betting scraper for League of Legends matches.
    Fetches rendered page content using Playwright and parses betting data.
    """

    # Try multiple potential LoL betting URLs on STS
    BASE_URLS = [
        "https://www.sts.pl/zaklady-bukmacherskie/esport/league-of-legends/156/992",
        "https://www.sts.pl/zaklady-bukmacherskie/esport/league-of-legends",
        "https://www.sts.pl/zaklady-bukmacherskie/esport/league-of-legends/lec",
        "https://www.sts.pl/zaklady-bukmacherskie/esport",
    ]

    def __init__(self):
        """Initialize STS betting scraper."""
        self._last_working_url = self.BASE_URLS[0]

    def get_html_content(self) -> Optional[str]:
        """
        Fetch the rendered HTML content of the LoL betting page using Playwright.
        Tries multiple URLs to find League of Legends content.

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

                # Try each URL until we find one with LoL content or matches
                for base_url in self.BASE_URLS:
                    try:
                        page = browser.new_page(
                            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        )
                        # Add headers to look like a real browser
                        page.set_extra_http_headers({
                            "Accept-Language": "en-US,en;q=0.9",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        })

                        # Navigate to page and wait for content to load
                        print(f"  [STS] Navigating to {base_url}...")
                        page.goto(base_url, wait_until="load", timeout=20000)
                        # Wait additional time for content rendering
                        page.wait_for_timeout(2000)

                        # Scroll to bottom to load all lazy-loaded content
                        print(f"  [STS] Scrolling page to load all matches...")
                        page.evaluate("""() => {
                            window.scrollTo(0, document.body.scrollHeight);
                        }""")
                        page.wait_for_timeout(3000)

                        # Scroll back to top and wait for any additional rendering
                        page.evaluate("""() => {
                            window.scrollTo(0, 0);
                        }""")
                        page.wait_for_timeout(1000)

                        # Get the rendered HTML
                        html_content = page.content()

                        # Quick check: does this page have match structure?
                        if 'match-tile' in html_content:
                            # Store the URL that worked for reference
                            self._last_working_url = base_url
                            browser.close()
                            return html_content

                        page.close()
                    except Exception as url_error:
                        # Try next URL
                        continue

                browser.close()
                print(f"⚠ Could not fetch LoL content from any STS URL")
                return None

        except Exception as e:
            print(f"Failed to fetch HTML content: {e}")
            return None

    def get_matches_structured(self) -> list[BettingMatch]:
        """
        Get all League of Legends matches as structured dataclass objects.

        Returns:
            List of BettingMatch dataclass instances
        """
        html_content = self.get_html_content()
        if not html_content:
            return []
        
        return self.parse_matches_from_html(html_content)

    def parse_matches_from_html(self, html_content: str) -> list[BettingMatch]:
        """
        Parse League of Legends matches from HTML content.

        Args:
            html_content: Raw HTML from STS betting page

        Returns:
            List of BettingMatch objects
        """
        # Use XPath-based parsing first
        matches = self._parse_with_xpath(html_content)

        # If no matches found, try original separator-based parsing
        if not matches:
            soup = BeautifulSoup(html_content, "lxml")
            page_text = soup.get_text()
            matches = self._parse_with_separator(page_text)

        # If still no matches, try regex-based parsing (fallback)
        if not matches:
            soup = BeautifulSoup(html_content, "lxml")
            page_text = soup.get_text()
            matches = self._parse_with_regex(page_text)

        # Deduplicate matches based on team names and start time
        if matches:
            seen = set()
            unique_matches = []
            for m in matches:
                # Use a tuple of (team1, team2, start_time) as a unique identifier
                identifier = (
                    m.team1.lower().strip(),
                    m.team2.lower().strip(),
                    m.start_time.strftime("%Y-%m-%d %H:%M")
                )
                if identifier not in seen:
                    seen.add(identifier)
                    unique_matches.append(m)
            
            return unique_matches

        return matches

    def _parse_with_xpath(self, html_content: str) -> list[BettingMatch]:
        """
        Parse matches using BeautifulSoup with CSS selectors targeting match-tile divs.
        Handles both old (div-based) and new (span-based) HTML structures.
        """
        soup = BeautifulSoup(html_content, "lxml")
        matches = []
        match_id = 1

        # Common LoL team names and keywords to filter
        lol_keywords = {
            'fnatic', 'g2', 'sk', 'vitality', 'misfits', 'mad',
            'roccat', 'rge', 'astralis', 'excel', 'kcorp', 'koi',
            'los ratones', 'heretics', 'tes', 'invictus', 'top esports',
            'edg', 'lpl', 'lck', 'geng', 't1', 'damwon', 'griffin',
            'sandbox', 'hle', 'hanwha', 'lol', 'league of legends',
        }

        # Try both old and new HTML structures
        new_style_tiles = soup.find_all("div", class_=lambda x: x and 'one-ticket-match-tile' in x and 'header' not in x)
        old_style_tiles = soup.find_all("div", class_=lambda x: x and 'match-tile' in x)
        all_tiles = new_style_tiles if new_style_tiles else old_style_tiles

        for tile in all_tiles:
            try:
                tile_text = tile.get_text(" ", strip=True)
                is_lol = any(keyword.lower() in tile_text.lower() for keyword in lol_keywords)
                if not is_lol:
                    continue

                team_spans = tile.find_all("span")
                teams = []
                if team_spans:
                    for span in team_spans:
                        text = span.get_text(strip=True)
                        if text and text != '-' and text != '+' and 1 < len(text) < 50:
                            if not text.isdigit() and not text.replace(',', '').replace('.', '').isdigit():
                                teams.append(text)

                if not teams:
                    team_divs = tile.find_all("div", class_=lambda x: x and 'teams__team' in x)
                    teams = [d.get_text(strip=True) for d in team_divs if d.get_text(strip=True) != '-']

                valid_teams = [t for t in teams if t and len(t) > 1 and t != '-']
                if len(valid_teams) < 2:
                    continue

                team1_text = valid_teams[0]
                team2_text = valid_teams[1]

                time_match = re.search(r'(\d{1,2}):(\d{2})', tile_text)
                if not time_match:
                    continue
                hour, minute = int(time_match.group(1)), int(time_match.group(2))

                date_match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', tile_text)
                if date_match:
                    day, month, year = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                    start_datetime = datetime(year, month, day, hour, minute)
                else:
                    now = datetime.now()
                    start_datetime = datetime.combine(now.date(), datetime.strptime(f"{hour}:{minute}", "%H:%M").time())
                    if start_datetime < now:
                        start_datetime += timedelta(days=1)

                market_odds = [
                    BettingOdds(outcome=team1_text, odds=1.5, is_active=True),
                    BettingOdds(outcome=team2_text, odds=1.5, is_active=True)
                ]

                market = BettingMarket(
                    market_type="match_winner",
                    market_type_id=None,
                    odds_list=market_odds,
                    description="Match Winner",
                )

                betting_match = BettingMatch(
                    match_id=match_id,
                    team1=team1_text,
                    team2=team2_text,
                    tournament="League of Legends",
                    start_time=start_datetime,
                    is_live=False,
                    markets={"match_winner": market},
                    match_url=self._last_working_url,
                )

                matches.append(betting_match)
                match_id += 1
            except Exception:
                continue

        return matches

    def _parse_with_separator(self, page_text: str) -> list[BettingMatch]:
        """Parse using '-' separator pattern."""
        lines = [line.strip() for line in page_text.split("\n") if line.strip()]
        matches = []
        i = 0
        while i < len(lines):
            if lines[i] == "-" and 0 < i < len(lines) - 1:
                team1 = lines[i - 1]
                team2 = lines[i + 1]
                start_time = None
                team1_odds = None
                team2_odds = None
                for j in range(i + 2, min(i + 20, len(lines))):
                    line = lines[j]
                    if re.match(r"\d{1,2}\.\d{1,2}\.\d{4}", line):
                        dt_match = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4}),\s*(\d{1,2}):(\d{2})", line)
                        if dt_match:
                            day, month, year, hour, minute = dt_match.groups()
                            start_time = f"{year}-{month.zfill(2)}-{day.zfill(2)}T{hour.zfill(2)}:{minute.zfill(2)}:00"
                    if line == "1" and j + 1 < len(lines):
                        try:
                            val = lines[j + 1].strip()
                            if re.match(r"^\d+\.\d{2}$", val): team1_odds = float(val)
                        except (ValueError, IndexError): pass
                    if line == "2" and j + 1 < len(lines):
                        try:
                            val = lines[j + 1].strip()
                            if re.match(r"^\d+\.\d{2}$", val): team2_odds = float(val)
                        except (ValueError, IndexError): pass
                    if (line == "-" or line.startswith("League of Legends")) and j > i + 5:
                        break
                if start_time and team1_odds and team2_odds:
                    odds_list = [
                        BettingOdds(outcome=team1, odds=team1_odds, is_active=True),
                        BettingOdds(outcome=team2, odds=team2_odds, is_active=True),
                    ]
                    market = BettingMarket(
                        market_type="match_winner",
                        market_type_id=None,
                        odds_list=odds_list,
                        description="Match Winner",
                    )
                    match = BettingMatch(
                        match_id=len(matches) + 1,
                        team1=team1,
                        team2=team2,
                        tournament="League of Legends",
                        start_time=datetime.fromisoformat(start_time),
                        is_live=False,
                        markets={"match_winner": market},
                        match_url=self.BASE_URLS[0],
                    )
                    matches.append(match)
            i += 1
        return matches

    def _parse_with_regex(self, page_text: str) -> list[BettingMatch]:
        """Parse using regex pattern matching fallback."""
        matches = []
        match_id = 1
        time_pattern = r"(\w+(?:\s+\w+)*?)\s{2,}(\d{1,2}):(\d{2})\s*[-–]\s*(\w+(?:\s+\w+)?).*?(\d+\.\d{2}).*?(\d+\.\d{2})"
        for match_obj in re.finditer(time_pattern, page_text):
            try:
                team1 = match_obj.group(1).strip()
                hour, minute = int(match_obj.group(2)), int(match_obj.group(3))
                team2 = match_obj.group(4).strip()
                odd1, odd2 = float(match_obj.group(5)), float(match_obj.group(6))
                if len(team1) < 2 or len(team2) < 2: continue
                time_obj = datetime.strptime(f"{hour}:{minute}", "%H:%M").time()
                now = datetime.now()
                start_datetime = datetime.combine(now.date(), time_obj)
                if start_datetime < now: start_datetime += timedelta(days=1)
                market_odds = [
                    BettingOdds(outcome=team1, odds=odd1, is_active=True),
                    BettingOdds(outcome=team2, odds=odd2, is_active=True)
                ]
                market = BettingMarket(market_type="match_winner", market_type_id=None, odds_list=market_odds, description="Match Winner")
                betting_match = BettingMatch(
                    match_id=match_id, team1=team1, team2=team2, tournament="League of Legends",
                    start_time=start_datetime, is_live=False, markets={"match_winner": market}, match_url=self.BASE_URLS[0]
                )
                matches.append(betting_match)
                match_id += 1
            except Exception: continue
        return matches

    def get_match_moneyline(self, match: BettingMatch) -> Optional[BettingMarket]:
        """Extract moneyline odds."""
        for key in ["match_winner", "match_winner_main", "match_winner_alt"]:
            if key in match.markets: return match.markets[key]
        return None

    def get_match_handicaps(self, match: BettingMatch) -> list:
        """Extract handicaps."""
        return [m for n, m in match.markets.items() if any(k in n for k in ["handicap", "over_under", "spread"])]
