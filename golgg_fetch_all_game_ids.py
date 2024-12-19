from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from src.utils.scrapers.golgg import GOLGG_TOURNAMENT_API, GOLGG_URL, GolggScraper
from tqdm.asyncio import tqdm_asyncio
import json


async def main():
    async with GolggScraper() as scraper:
        tournaments = await scraper.get_tournaments_in_season(10)

        # Step 1: Extract matches from tournaments
        all_games = set()
        print("Extracting matches from tournaments...")
        match_tasks = [
            scraper.get_matches_in_tournament(tournament["trname"])
            for tournament in tournaments
        ]
        matches_by_tournament = await tqdm_asyncio.gather(
            *match_tasks, desc="Tournaments", leave=True
        )

        # Flatten the matches list
        all_matches = {match for matches in matches_by_tournament for match in matches}

        # Step 2: Extract games from matches
        print("Extracting games from matches...")
        game_tasks = [scraper.get_games_ids_in_match(match) for match in all_matches]
        games_by_match = await tqdm_asyncio.gather(
            *game_tasks, desc="Matches", leave=True
        )

        # Combine all game IDs
        for games in games_by_match:
            all_games.update(games)

    games = list(all_games)
    print(f"Found {len(games)} games.")
    with open("games.json", "w") as f:
        json.dump(games, f, indent=4)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
