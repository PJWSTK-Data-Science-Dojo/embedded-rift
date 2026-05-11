import json
import asyncio
import argparse
import sys
from pathlib import Path
from tqdm.asyncio import tqdm

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from utils.scrapers.golgg import GolggScraper

CONCURRENCY_LIMIT = 60
TOURNAMENT_CONCURRENCY_LIMIT = 20
MAX_RETRIES = 5


async def get_tournament_matches(
    scraper: GolggScraper, tournaments: list[dict]
) -> list[dict]:
    all_matches = []
    semaphore = asyncio.Semaphore(TOURNAMENT_CONCURRENCY_LIMIT)

    async def fetch_tournament(tournament: dict) -> list[dict]:
        tournament_name = tournament.get("trname")
        if not tournament_name:
            return []
        async with semaphore:
            return await scraper.get_matches_in_tournament(tournament_name)

    tasks = [fetch_tournament(tournament) for tournament in tournaments]
    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Fetching matches for tournaments"
    ):
        all_matches.extend(await coro)
    return all_matches


async def fetch_match_games(
    scraper: GolggScraper, match: dict, semaphore: asyncio.Semaphore
) -> list[dict]:
    match_id = match.get("match_id")
    if not match_id:
        print("Match ID not found, skipping...")
        return []

    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            try:
                return await scraper.get_games_in_match(match_id)
            except Exception as e:
                print(f"[!][Match {match_id}] attempt {attempt} failed: {e}")
        await asyncio.sleep(attempt)  # simple back-off before retrying

    print(f"[!][Match {match_id}] giving up after {MAX_RETRIES} attempts")
    return []


async def get_games(
    matches, scraper: GolggScraper, concurrency: int = CONCURRENCY_LIMIT
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    all_games = []

    tasks = [fetch_match_games(scraper, match, semaphore) for match in matches]

    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Fetching games"
    ):
        result = await coro
        all_games.extend(result)

    return all_games


async def fetch_match_with_games(
    scraper: GolggScraper, match: dict, semaphore: asyncio.Semaphore
) -> dict:
    """Fetch games for one match and return a match document with nested games."""
    match_doc = dict(match)
    games = await fetch_match_games(scraper, match, semaphore)
    for game in games:
        game["patch"] = match_doc.get("patch")
        game["date"] = match_doc.get("date")
        game["tournament_name"] = match_doc.get("tournament_name")
    match_doc["games"] = games
    enrich_match_with_nested_game_metadata(match_doc)
    return match_doc


async def get_matches_with_games(
    matches: list[dict], scraper: GolggScraper, concurrency: int = CONCURRENCY_LIMIT
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [fetch_match_with_games(scraper, match, semaphore) for match in matches]
    result = []

    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Fetching match games"
    ):
        result.append(await coro)

    return result


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [strip_redundant_player_maps(item) for item in data]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def strip_redundant_player_maps(match: dict) -> dict:
    """Keep player data only in t1_players/t2_players nested in games."""
    cleaned = dict(match)
    redundant_keys = {
        "t1_player_ids",
        "t2_player_ids",
        "t1_player_names",
        "t2_player_names",
        "t1_champions",
        "t2_champions",
    }
    for key in redundant_keys:
        cleaned.pop(key, None)

    cleaned_games = []
    for game in cleaned.get("games") or []:
        cleaned_game = dict(game)
        for key in redundant_keys:
            cleaned_game.pop(key, None)
        cleaned_games.append(cleaned_game)
    if "games" in cleaned:
        cleaned["games"] = cleaned_games
    return cleaned


def deduplicate_by_key(items: list[dict], key: str) -> list[dict]:
    result: dict[str, dict] = {}
    without_key = []
    for item in items:
        value = item.get(key)
        if value is None:
            without_key.append(item)
            continue
        result[str(value)] = item
    return without_key + list(result.values())


def count_nested_games_by_tournament(matches: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in matches:
        tournament = match.get("tournament_name") or match.get("tournament")
        if not tournament:
            continue
        counts[tournament] = counts.get(tournament, 0) + len(match.get("games") or [])
    return counts


def count_nested_games_by_match(matches: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in matches:
        match_id = match.get("match_id")
        if not match_id:
            continue
        counts[str(match_id)] = len(match.get("games") or [])
    return counts


def merge_match_metadata(existing: dict, fresh: dict) -> dict:
    """Update match-level fields while preserving already fetched nested games."""
    games = existing.get("games") or []
    merged = dict(existing)
    merged.update(fresh)
    merged["games"] = games
    enrich_match_with_nested_game_metadata(merged)
    return merged


def normalize_name(name: str | None) -> str:
    return " ".join((name or "").lower().split())


def infer_best_of_from_match(match: dict) -> int | None:
    try:
        t1_score = int(match.get("t1_score", 0))
        t2_score = int(match.get("t2_score", 0))
    except (TypeError, ValueError):
        return None

    if t1_score == t2_score:
        games_played = t1_score + t2_score
        return games_played if games_played > 0 else None

    wins_needed = max(t1_score, t2_score)
    if wins_needed <= 0:
        return None
    return (wins_needed * 2) - 1


def enrich_match_result_flags(match: dict) -> None:
    try:
        t1_score = int(match.get("t1_score", 0))
        t2_score = int(match.get("t2_score", 0))
    except (TypeError, ValueError):
        return

    match["games_played"] = t1_score + t2_score
    match["t1_win"] = t1_score > t2_score
    match["t2_win"] = t2_score > t1_score
    match["draw"] = t1_score == t2_score
    match["best_of"] = infer_best_of_from_match(match)


def enrich_match_with_nested_game_metadata(match: dict) -> None:
    """Add match-level IDs/BO derived from match score and nested games."""
    enrich_match_result_flags(match)

    games = match.get("games") or []
    if not games:
        return

    first_game = games[0]
    game_teams = {
        normalize_name(first_game.get("t1_name")): {
            "team_id": first_game.get("t1_id"),
            "players": first_game.get("t1_players") or {},
        },
        normalize_name(first_game.get("t2_name")): {
            "team_id": first_game.get("t2_id"),
            "players": first_game.get("t2_players") or {},
        },
    }

    t1_info = game_teams.get(normalize_name(match.get("sname_t1")))
    t2_info = game_teams.get(normalize_name(match.get("sname_t2")))

    if t1_info:
        match["t1_id"] = t1_info.get("team_id")
    if t2_info:
        match["t2_id"] = t2_info.get("team_id")


def expected_games_for_match(match: dict) -> int:
    try:
        return int(match.get("t1_score", 0)) + int(match.get("t2_score", 0))
    except (ValueError, TypeError):
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch GOL.gg matches with nested game details"
    )
    parser.add_argument(
        "--output-path",
        "--matches-path",
        dest="output_path",
        type=Path,
        default=Path("data/golgg_matches.json"),
        help="JSON output path. The file contains match objects with nested games.",
    )
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY_LIMIT,
        help="How many matches to fetch concurrently.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="How many match documents to fetch before saving progress.",
    )
    parser.add_argument("--refresh-matches", action="store_true", help="Fetch match lists for all tournaments instead of only tournaments with missing games")
    parser.add_argument(
        "--refetch-games",
        action="store_true",
        help="Fetch games again even for matches that already have the expected number of games.",
    )
    parser.add_argument(
        "--match-id",
        help="Fetch/refetch only one existing or newly discovered match id.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    output_path = args.output_path

    print(f"Loading existing match documents from {output_path}...")
    matches = deduplicate_by_key(load_json_list(output_path), "match_id")

    # Count games per tournament and per match
    games_per_tournament = count_nested_games_by_tournament(matches)
    games_per_match = count_nested_games_by_match(matches)

    async with GolggScraper(max_pages=args.max_pages) as scraper:
        print("Fetching all tournaments from GOL.gg...")
        all_tournaments = await scraper.get_all_tournaments()

        tournaments_to_fetch = []
        for t in all_tournaments:
            name = t.get("trname")
            if not name:
                continue
            try:
                nbgames = int(t["nbgames"])
            except (ValueError, TypeError):
                nbgames = 0

            have_games = games_per_tournament.get(name, 0)
            if args.refresh_matches or have_games < nbgames:
                tournaments_to_fetch.append(t)

        if tournaments_to_fetch:
            print(
                f"Found {len(tournaments_to_fetch)} tournaments with potentially missing games."
            )
            new_matches_list = await get_tournament_matches(scraper, tournaments_to_fetch)

            # Update matches.json
            new_match_count = 0
            updated_match_count = 0

            # Create a map for existing matches for easy update
            existing_matches_map = {
                str(m["match_id"]): i for i, m in enumerate(matches) if m.get("match_id")
            }

            for m in new_matches_list:
                raw_mid = m.get("match_id")
                if not raw_mid:
                    continue
                mid = str(raw_mid)
                if mid not in existing_matches_map:
                    matches.append(m)
                    existing_matches_map[mid] = len(matches) - 1
                    new_match_count += 1
                else:
                    # Update existing match info (e.g. score might have changed)
                    idx = existing_matches_map[mid]
                    if matches[idx] != m:
                        matches[idx] = merge_match_metadata(matches[idx], m)
                        updated_match_count += 1

            if new_match_count > 0 or updated_match_count > 0:
                print(
                    f"Added {new_match_count} new matches and updated {updated_match_count} matches in {output_path}"
                )
                matches = deduplicate_by_key(matches, "match_id")
                save_json(output_path, matches)
        else:
            print("No new tournaments or matches found.")

        # Identify missing or incomplete matches
        missing_matches = []
        for m in matches:
            mid = m.get("match_id")
            if not mid:
                continue
            mid = str(mid)
            if args.match_id and mid != str(args.match_id):
                continue
            expected_games = expected_games_for_match(m)
            if expected_games <= 0:
                continue

            have_games = games_per_match.get(mid, 0)
            if args.refetch_games or have_games < expected_games:
                missing_matches.append(m)

        if not missing_matches:
            print("All matches already fetched.")
            return

        print(
            f"Found {len(missing_matches)} missing or incomplete matches. Fetching games..."
        )

        batch_size = max(1, args.batch_size)
        matches_by_id = {str(m["match_id"]): m for m in matches if m.get("match_id")}
        fetched_total = 0

        for start in range(0, len(missing_matches), batch_size):
            batch = missing_matches[start : start + batch_size]
            print(
                f"Fetching batch {start // batch_size + 1}: "
                f"{start + 1}-{start + len(batch)} / {len(missing_matches)}"
            )
            fetched_match_docs = await get_matches_with_games(
                batch, scraper, concurrency=args.concurrency
            )
            fetched_match_docs = [
                m for m in fetched_match_docs if m.get("match_id") and m.get("games")
            ]

            for match_doc in fetched_match_docs:
                matches_by_id[str(match_doc["match_id"])] = match_doc

            if fetched_match_docs:
                fetched_total += len(fetched_match_docs)
                matches = deduplicate_by_key(list(matches_by_id.values()), "match_id")
                total_games = sum(len(m.get("games") or []) for m in matches)
                print(
                    f"Saving progress: {len(matches)} matches, "
                    f"{total_games} nested games to {output_path}"
                )
                save_json(output_path, matches)

        if fetched_total == 0:
            print("No match games fetched successfully; keeping existing file unchanged.")
            return

        print(f"Finished fetching games for {fetched_total} matches.")


if __name__ == "__main__":
    asyncio.run(main())
