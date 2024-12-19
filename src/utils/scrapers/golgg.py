from typing import Self
from playwright.async_api import async_playwright, Page
import httpx
from playwright.async_api import ElementHandle
import re
from tqdm import tqdm
from urllib.parse import urljoin, quote
import asyncio
import time
import parsel

GOLGG_URL = "https://gol.gg"
GOLGG_TOURNAMENT_API = "https://gol.gg/tournament/ajax.trlist.php"


class GolggScraper:
    def __init__(self, max_pages: int = 20):
        self.semaphore = asyncio.Semaphore(max_pages)

    async def start(self, headless: bool = True) -> Self:
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
            }
        )
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=headless)
        return self

    async def stop(self):
        await self.browser.close()
        await self.playwright.stop()
        if self.client:
            await self.client.aclose()

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def click_consent(self, page: Page):
        # Example if consent handling is needed
        buttons = await page.query_selector_all("button.fc-button")
        if len(buttons) > 1:
            await buttons[1].click()

    async def get_tournaments_in_season(self, season: int = 9) -> list[dict]:
        response = await self.client.post(
            GOLGG_TOURNAMENT_API,
            data={"season": f"S{season}"},
        )
        data = response.json()
        return data

    async def get_all_tournaments(self) -> list[dict]:
        result = []
        for season in range(9, 15):
            data = await self.get_tournaments_in_season(season)
            result.extend(data)
        return result

    async def get_matches_in_tournament(self, tournament_name: str) -> set[str]:
        """Process a single tournament using a dedicated page."""
        matches = set()
        encoded_trname = quote(tournament_name)
        s = f"tournament-matchlist/{encoded_trname}/"

        url = f"{GOLGG_URL}/tournament/{s}"

        response = await self.client.get(url)
        html = response.content.decode("utf-8")

        sel = parsel.Selector(text=html)
        rows = sel.css(".table_list tbody tr")

        # Extract links and match IDs
        matches = set()
        for row in rows:
            href = row.css("a::attr(href)").get()

            if not href:
                continue

            # Extract match ID using regex
            pattern = r"stats/(\d+)/"
            match = re.search(pattern, href)
            if not match:
                print("Couldn't extract match id from", href)
                continue

            match_id = match.group(1)
            matches.add(match_id)

        return matches

    async def get_games_ids_in_match(self, match_id):
        """Get the games ids in a match."""
        s = f"/game/stats/{match_id}/page-summary/"
        url = f"{GOLGG_URL}{s}"
        async with self.semaphore:
            response = await self.client.get(url)

        if response.status_code != 200:
            print("Couldn't fetch", url)
            return set()

        html = response.content.decode("utf-8")
        with open("test.html", "w") as f:
            f.write(html)
        sel = parsel.Selector(text=html)
        navbar = sel.css("#gameMenuToggler")
        if not navbar:
            print("Couldn't find navbar in", url)
            return set()

        links = navbar[0].css("a")
        # Extract game IDs
        games = set()
        for link in links:
            text = link.css("::text").get()
            if text and text.startswith("GAME"):
                continue

            href = link.css("::attr(href)").get()
            if not href:
                print("Couldn't extract href from", link.get())
                continue

            # Extract game ID using regex
            pattern = r"stats/(\d+)/"
            match = re.search(pattern, href)

            if not match:
                print("Couldn't extract game id from", href)
                continue

            game_id = match.group(1)
            games.add(game_id)

        return games
