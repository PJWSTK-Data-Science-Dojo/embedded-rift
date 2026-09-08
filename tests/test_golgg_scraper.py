"""Regression fixtures from GOL.GG; no network requests or operational database."""

import asyncio
import importlib.util
from pathlib import Path

import httpx
from parsel import Selector
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("historical_golgg_scraper", ROOT / "src/utils/scrapers/golgg.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
GolggScraper = MODULE.GolggScraper
FIXTURES = Path(__file__).parent / "fixtures/golgg"


def test_historical_scores_follow_team_identity_not_winner_first():
    async def run():
        scraper = GolggScraper()
        html = (FIXTURES / "historical_result_rows.html").read_text()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html))) as client:
            scraper.client = client
            rows = await scraper.get_matches_in_tournament("Historical fixture")
        by_id = {str(row["match_id"]): row for row in rows}
        assert (by_id["325"]["t1_score"], by_id["325"]["t2_score"]) == (0, 1)
        assert (by_id["1077"]["t1_score"], by_id["1077"]["t2_score"]) == (2, 0)
        assert by_id["1081"]["score_alignment"] == "unresolved"
        assert (by_id["1081"]["result_left_team"], by_id["1081"]["result_right_team"]) == ("Team Vulcun", "Gambit Gaming")
    asyncio.run(run())


def test_duplicate_team_ids_cannot_silently_overwrite_opponent_roster():
    scraper = GolggScraper()
    source = Selector(text=(FIXTURES / "duplicate_team_game.html").read_text())
    with pytest.raises(ValueError, match="(?i)(duplicate|distinct|same team)"):
        asyncio.run(scraper.get_players_in_game(source))


def test_preview_without_result_is_unplayed_but_broken_completed_result_is_not():
    async def run():
        html = '<table class="table_list" data-sort="0"><tbody>'
        for mid, page in (("1", "page-preview"), ("2", "page-summary")):
            html += f'<tr><td><a href="../game/stats/{mid}/{page}/">A vs B</a></td><td>A</td><td> - </td><td>B</td></tr>'
        html += '</tbody></table>'
        scraper = GolggScraper()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html))) as client:
            scraper.client = client
            assert await scraper.get_matches_in_tournament("Fixture") == []
        assert {row["match_id"] for row in scraper.unplayed_rows} == {"1"}
        assert {row["match_id"] for row in scraper.failed_rows} == {"2"}
    asyncio.run(run())


def test_missing_tournament_table_is_not_a_successful_empty_history():
    async def run():
        scraper = GolggScraper()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>Unavailable</html>"))) as client:
            scraper.client = client
            with pytest.raises(ValueError, match="games table"):
                await scraper.get_matches_in_tournament("Fixture")
    asyncio.run(run())
