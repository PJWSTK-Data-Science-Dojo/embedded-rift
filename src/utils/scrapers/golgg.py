from typing import Self
from playwright.async_api import async_playwright, Page
import httpx
from playwright.async_api import ElementHandle
import re
from tqdm import tqdm
from urllib.parse import urljoin, quote
import asyncio
import time

GOLGG_URL = "https://gol.gg"
GOLGG_TOURNAMENT_API = "https://gol.gg/tournament/ajax.trlist.php"


class GolggScraper:
    def __init__(self, max_pages: int = 20):
        self.semaphore = asyncio.Semaphore(max_pages)

    async def start(self, headless: bool = True) -> Self:
        self.client = httpx.AsyncClient()
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
        url = GOLGG_TOURNAMENT_API
        async with httpx.AsyncClient() as client:
            response = await self.client.post(
                "https://gol.gg/tournament/ajax.trlist.php",
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
        async with self.semaphore:
            page = await self.browser.new_page()
            try:
                url = f"{GOLGG_URL}/tournament/{s}"
                await page.goto(url)
                await self.click_consent(page)
                table = await page.query_selector(".table_list")
                tbody = await table.query_selector("tbody")
                rows = await tbody.query_selector_all("tr")
                for row in rows:
                    link = await row.query_selector("a")
                    href = await link.get_attribute("href")
                    if not href:
                        continue

                    pattern = r"stats/(\d+)/"
                    match: re.Match = re.search(pattern, href)

                    if not match:
                        print("Couldnt extract match id from", href)
                        continue

                    match_id = match.group(1)
                    matches.add(match_id)

            finally:
                await page.close()

        return matches

    async def get_games_ids_in_match(self, match_id):
        """Get the games ids in a match."""
        s = f"/game/stats/{match_id}/page-summary/"

        async with self.semaphore:
            page = await self.browser.new_page()
            try:
                await page.goto(f"{GOLGG_URL}{s}")
                await self.click_consent(page)
                navbar = await page.query_selector("#gameMenuToggler")
                links = await navbar.query_selector_all("a")
                games = set()

                for link in links:
                    if (await link.text_content()).startswith("GAME"):
                        continue

                    href = await link.get_attribute("href")
                    if not href:
                        print("Couldnt extract href from", link.inner_html())
                        continue

                    pattern = r"stats/(\d+)/"
                    match: re.Match = re.search(pattern, href)

                    if not match:
                        print("Couldnt extract game id from", href)
                        continue

                    game_id = match.group(1)
                    games.add(game_id)
            finally:
                await page.close()

        return games
