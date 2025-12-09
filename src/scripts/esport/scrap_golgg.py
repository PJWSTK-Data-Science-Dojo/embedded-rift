from utils.scrapers.golgg import GolggScraper
import json
import asyncio
import statistics
from pathlib import Path
from tqdm.asyncio import tqdm
from tqdm import tqdm as stqdm

CONCURRENCY_LIMIT = 15
MAX_RETRIES = 5


async def get_tournament_matches(
    scraper: GolggScraper, tournaments: list[str]
) -> list[dict]:
    all_matches = []
    for tournament in tqdm(tournaments):
        tournament_name = tournament["trname"]
        matches = await scraper.get_matches_in_tournament(tournament_name)
        all_matches.extend(matches)
    return all_matches


async def fetch_match_games(
    scrapper: GolggScraper, match: dict, semaphore: asyncio.Semaphore
) -> list[dict]:
    match_id = match.get("match_id")
    if not match_id:
        print("Match ID not found, skipping...")
        return []

    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            try:
                return await scrapper.get_games_in_match(match_id)
            except Exception as e:
                print(f"[!][Match {match_id}] attempt {attempt} failed:", e)
        await asyncio.sleep(attempt)  # simple back-off before retrying

    print(f"[!][Match {match_id}] giving up after {MAX_RETRIES} attempts")
    return []


async def get_games(matches, games_path: Path) -> list[dict]:
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    all_games = []

    async with GolggScraper(max_pages=40) as scrapper:
        tasks = [fetch_match_games(scrapper, match, semaphore) for match in matches]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await coro
            all_games.extend(result)

    with open(games_path, "w") as f:
        json.dump(all_games, f, indent=4)

    return all_games


async def main():
    tournaments_path = Path("data/tournaments.json")
    matches_path = Path("data/matches.json")
    games_path = Path("data/games.json")
    # if tournaments_path.exists():
    #     print("Tournaments already downloaded, skipping...")
    #     return
    # tournaments = []
    # matches = []
    # if tournaments_path.exists():
    #     with open(tournaments_path, "r") as f:
    #         tournaments = json.load(f)

    # if matches_path.exists():
    #     with open(matches_path, "r") as f:
    #         matches = json.load(f)

    # if not tournaments and not matches:
    async with GolggScraper(max_pages=3) as scrapper:
        tournaments = await scrapper.get_tournaments_in_season(15)

        mean = statistics.mean([int(t["nbgames"]) for t in tournaments])
        median = statistics.median([int(t["nbgames"]) for t in tournaments])
        accumulated = sum([int(t["nbgames"]) for t in tournaments])
        minimum = min([int(t["nbgames"]) for t in tournaments])
        maximum = max([int(t["nbgames"]) for t in tournaments])
        is_nan = any([int(t["nbgames"]) == float("nan") for t in tournaments])

        print("Tournaments saved to tournaments.json")
        print("Total tournaments:", len(tournaments))
        print("Mean number of games:", mean)
        print("Median number of games:", median)
        print("Total number of games:", accumulated)
        print("Minimum number of games:", minimum)
        print("Maximum number of games:", maximum)
        print("Contains NaN:", is_nan)

        matches = await get_tournament_matches(scrapper, tournaments=tournaments)
        with open(matches_path, "w") as f:
            json.dump(matches, f, indent=4)

        print("Matches saved to matches.json")
        print("Total matches:", len(matches))

    saved_games = []
    if games_path.exists():
        with open(games_path, "r") as f:
            saved_games = json.load(f)

    saved_match_ids = {game["match_id"] for game in saved_games}

    matches = [m for m in matches if m["match_id"] not in saved_match_ids]

    games = await get_games(matches, games_path)

    updated_games = []
    for match in stqdm(matches):
        new_mgames = []
        for game in games:
            if game["match_id"] != match["match_id"]:
                continue

            game["date"] = match["date"]

            game["tournament"] = match["tournament_name"]
            new_mgames.append(game)

        updated_games.extend(new_mgames)
    saved_games.extend(updated_games)
    with open(games_path, "w") as f:
        json.dump(saved_games, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
