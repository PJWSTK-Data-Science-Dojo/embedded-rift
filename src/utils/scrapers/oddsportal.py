import json
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

ODDS_PORTAL_URL = "https://www.oddsportal.com/pl/esports/results/"


class OddsPortalScraper:
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

    async def get_all_tournaments(self) -> list[dict]:

        page = await self.browser.new_page()
        await page.goto(ODDS_PORTAL_URL)
        content = await page.content()
        selector = parsel.Selector(content)
        # Find all a link that contains "league-of-legends" in href
        links = selector.css("a[href*='league-of-legends/league-of-legends']")
        tournaments = []
        for link in links:
            href = link.attrib["href"]
            name = link.css("::text").get()
            full_url = urljoin(ODDS_PORTAL_URL, href)
            tournaments.append({"name": name, "url": full_url})

        return tournaments

    async def get_tournament_matches(self, tournament_url: str) -> list[dict]:
        page = await self.browser.new_page()
        await page.goto(tournament_url)
        # Wait if There will be text "Niestety, nie mo" or Data rows
        selector = await page.wait_for_selector(".eventRow", timeout=5000)
        if not selector:
            return []

        content = await page.content()

        selector = parsel.Selector(content)
        # Example: Extract match data
        matches = []
        # Find all .eventRow > div with atribute data-testid="game-row"
        match_rows = selector.css(".eventRow")

        for row in match_rows:
            event_row = row.css("div[data-testid='game-row']")
            link = row.css("div[data-testid='game-row'] > a")
            href = link.attrib["href"]
            full_url = urljoin(ODDS_PORTAL_URL, href)
            team1_name, team2_name, *rest = event_row.css(
                ".participant-name::text"
            ).getall()
            print(team1_name, team2_name)
            # print(rest)
            team_1_score, team_2_score, *rest = participant_box = event_row.css(
                'div[data-testid="event-participants"] > div > div > div > div::text'
            ).getall()

            match_info = {
                "url": full_url,
                "team1": team1_name.strip(),
                "team2": team2_name.strip(),
                "team1_score": team_1_score.strip(),
                "team2_score": team_2_score.strip(),
            }
            print(match_info)
            matches.append(match_info)
        return matches


async def main():
    async with OddsPortalScraper() as scraper:
        tournaments = await scraper.get_all_tournaments()
        for tournament in tqdm(tournaments):
            print(tournament)
            matches = await scraper.get_tournament_matches(tournament["url"])
            break


if __name__ == "__main__":
    asyncio.run(main())
