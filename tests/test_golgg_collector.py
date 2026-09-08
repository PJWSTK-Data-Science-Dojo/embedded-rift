"""Collector contracts: complete maps, identity-safe labels, and durable resume."""
import asyncio
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("historical_golgg_collector", ROOT / "src/scripts/esport/scrape_golgg.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def game(game_id="1", blue="10", red="20", blue_win=True, best_of=1):
    return {
        "game_id": game_id, "match_id": "1", "t1_id": blue, "t2_id": red,
        "source_best_of": best_of,
        "t1_name": f"Team {blue}", "t2_name": f"Team {red}",
        "t1_side": "blue", "t2_side": "red", "t1_win": blue_win,
        "t2_win": not blue_win, "draw": False,
        "t1_players": {role: {"player_id": f"{blue}-{role}"} for role in MODULE.ROLES},
        "t2_players": {role: {"player_id": f"{red}-{role}"} for role in MODULE.ROLES},
    }


def header(score=(1, 0), names=("Team 10", "Team 20")):
    return {"match_id": "1", "sname_t1": names[0], "sname_t2": names[1],
            "t1_score": score[0], "t2_score": score[1], "date": "2020-01-01",
            "tournament_name": "Fixture", "source_result": {"scoreboard": "raw"}}


def test_rotating_blue_side_keeps_series_wins_attached_to_exact_ids():
    games = [game(best_of=3), game("2", "20", "10", True, best_of=3), game("3", "20", "10", False, best_of=3)]
    row = MODULE.reconcile_match(header((2, 1)), games)
    assert (row["t1_id"], row["t2_id"], row["t1_score"], row["t2_score"]) == ("10", "20", 2, 1)
    assert row["games"][1]["t1_id"] == "20"
    assert row["source_header"]["source_result"]["scoreboard"] == "raw"


def test_ambiguous_aliases_canonicalize_without_guessing_score_orientation():
    row = MODULE.reconcile_match(header((0, 1), ("Short A", "Short B")), [game()])
    assert (row["sname_t1"], row["sname_t2"], row["t1_score"], row["t2_score"]) == ("Team 10", "Team 20", 1, 0)
    assert row["source_header"]["t1_score"] == 0
    assert row["reconciliation"]["header_orientation"] == "ambiguous_aliases"


def test_resolved_header_opposing_winner_is_quarantinable_not_silently_inverted():
    with pytest.raises(ValueError, match="score"):
        MODULE.reconcile_match(header((0, 1)), [game()])


@pytest.mark.parametrize("damage", ["partial", "duplicate_game", "duplicate_team", "roster_overlap", "both_win", "wrong_match"])
def test_invalid_maps_never_enter_clean_dataset(damage):
    games = [game()]
    source = header()
    if damage == "partial":
        source = header((2, 0))
    elif damage == "duplicate_game":
        games.append(deepcopy(games[0]))
        source = header((2, 0))
    elif damage == "duplicate_team":
        games[0]["t2_id"] = "10"
    elif damage == "roster_overlap":
        games[0]["t2_players"] = deepcopy(games[0]["t1_players"])
    elif damage == "both_win":
        games[0]["t2_win"] = True
    elif damage == "wrong_match":
        games[0]["match_id"] = "999"
    with pytest.raises(ValueError):
        MODULE.reconcile_match(source, games)


def test_series_correction_requires_exact_snapshot_and_provenance():
    correction = {"series": {"1": {
        "expected": {"t1_name": "Team 10", "t2_name": "Team 20", "t1_score": 0, "t2_score": 1},
        "corrected": {"t1_id": "10", "t2_id": "20", "t1_score": 1, "t2_score": 0},
        "provenance": {"url": "https://example.org/verified-result"},
    }}}
    row = MODULE.reconcile_match(header((0, 1)), [game()], correction)
    assert row["t1_win"] is True
    correction["series"]["1"]["expected"]["t1_name"] = "Different team"
    with pytest.raises(ValueError, match="snapshot"):
        MODULE.reconcile_match(header((0, 1)), [game()], correction)

def test_source_listed_map_ids_must_match_the_complete_returned_order():
    first = game()
    first["source_game_ids"] = ["1", "2"]
    with pytest.raises(ValueError, match="source game IDs"):
        MODULE.reconcile_match(header(), [first])


def test_unresolved_scoreboard_orientation_is_not_mistaken_for_link_order():
    source = header((0, 1))
    source["score_alignment"] = "unresolved"
    row = MODULE.reconcile_match(source, [game()])
    assert row["t1_score"] == 1
    assert row["source_header"]["t1_score"] == 0


def test_journal_recovers_discovered_ids_even_before_snapshot_and_after_torn_tail(tmp_path):
    path = tmp_path / "fresh.json"
    store = MODULE.CollectionStore(path, resume=False, config={"scope": "fixture"})
    store.put(header(), "pending", ["awaiting maps"])
    with store.journal_path.open("ab") as handle:
        handle.write(b'{"op":"match"')
    resumed = MODULE.CollectionStore(path, resume=True, config={"scope": "fixture"})
    assert set(resumed.records) == {"1"}
    resumed.put(MODULE.reconcile_match(header(), [game()]), "clean", [])
    resumed.checkpoint("complete")
    assert json.loads(path.read_text())[0]["t1_win"] is True
    assert json.loads(resumed.quarantine_path.read_text()) == []
    assert json.loads(resumed.manifest_path.read_text())["counts"]["discovered_matches"] == 1


def test_existing_unowned_dataset_is_never_overwritten(tmp_path):
    path = tmp_path / "original.json"
    path.write_text('[{"old":true}]')
    with pytest.raises(ValueError, match="existing|owned"):
        MODULE.CollectionStore(path, resume=True, config={})
    assert path.read_text() == '[{"old":true}]'


def test_focused_refetch_quarantines_failure_without_reusing_stale_maps(tmp_path):
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps([header()]))
    output = tmp_path / "new.json"
    args = MODULE.parse_args(["--output-path", str(output), "--match-id", "1",
                              "--match-source", str(source_path), "--refetch-games"])

    class Scraper:
        async def get_games_in_match(self, match_id):
            raise ValueError("broken source identity")
        async def get_all_tournaments(self):
            raise AssertionError("focused fetch must not discover all tournaments")

    assert asyncio.run(MODULE.collect(args, Scraper(), {})) == 2
    assert json.loads(output.read_text()) == []
    unresolved = json.loads(output.with_suffix(".quarantine.json").read_text())
    assert unresolved[0]["match_id"] == "1"
    assert "broken source identity" in unresolved[0]["reasons"][0]
    assert not unresolved[0]["match"].get("games")


def test_refetch_generation_removes_previously_clean_maps_when_source_now_fails(tmp_path):
    source = tmp_path / "header.json"
    source.write_text(json.dumps(header()))
    output = tmp_path / "fresh.json"
    argv = ["--output-path", str(output), "--match-source", str(source),
            "--match-id", "1", "--refetch-games"]

    class Scraper:
        broken = False
        async def get_games_in_match(self, match_id):
            if self.broken:
                raise ValueError("fresh fetch failed")
            return [game()]

    scraper = Scraper()
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv), scraper, {})) == 0
    scraper.broken = True
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv + ["--resume", "--new-generation"]), scraper, {})) == 2
    assert json.loads(output.read_text()) == []
    unresolved = json.loads(output.with_suffix(".quarantine.json").read_text())
    assert unresolved[0]["match_id"] == "1"
    assert "games" not in unresolved[0]["match"]


def test_reconciliation_failure_retains_fresh_maps_in_quarantine(tmp_path):
    source = tmp_path / "header.json"
    source.write_text(json.dumps(header((2, 0))))
    output = tmp_path / "fresh.json"
    args = MODULE.parse_args(["--output-path", str(output), "--match-source", str(source), "--match-id", "1"])

    class Scraper:
        async def get_games_in_match(self, match_id):
            return [game()]

    assert asyncio.run(MODULE.collect(args, Scraper(), {})) == 2
    assert json.loads(output.read_text()) == []
    unresolved = json.loads(output.with_suffix(".quarantine.json").read_text())
    assert unresolved[0]["match"]["games"][0]["game_id"] == "1"


def test_worker_window_bounds_live_operations():
    async def run():
        active = 0
        peak = 0
        async def work(item):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return item * 2
        results = [result async for _, result, error in MODULE.bounded_results(range(30), work, 3) if error is None]
        assert sorted(results) == list(range(0, 60, 2))
        assert peak <= 3
    asyncio.run(run())


def test_focused_match_is_not_failed_by_unrelated_unplayed_rows(tmp_path):
    output = tmp_path / "focused.json"
    args = MODULE.parse_args([
        "--output-path", str(output), "--match-id", "1", "--tournament", "Fixture",
    ])

    class Scraper:
        failed_rows = [{
            "match_id": "2", "tournament_name": "Fixture",
            "error": "unplayed preview has no numeric score", "html": "<tr> - </tr>",
        }]

        async def get_matches_in_tournament(self, tournament):
            return [header()]

        async def get_games_in_match(self, match_id):
            return [game()]

    assert asyncio.run(MODULE.collect(args, Scraper(), {})) == 0
    assert {row["match_id"] for row in json.loads(output.read_text())} == {"1"}
    assert json.loads(output.with_suffix(".quarantine.json").read_text()) == []


def test_two_zero_bo2_remains_bo2_instead_of_becoming_bo3():
    maps = [game(best_of=2), game("2", best_of=2)]
    result = MODULE.reconcile_match(header((2, 0)), maps)
    assert result["best_of"] == 2


def test_partial_bo5_is_not_accepted_as_completed_bo3():
    maps = [game(best_of=5), game("2", best_of=5)]
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.reconcile_match(header((2, 0)), maps)


def test_maps_after_series_clincher_are_rejected():
    maps = [game(best_of=3), game("2", best_of=3), game("3", blue_win=False, best_of=3)]
    with pytest.raises(ValueError, match="completed series"):
        MODULE.reconcile_match(header((2, 1)), maps)


def test_successful_empty_season_has_no_missing_history(tmp_path):
    output = tmp_path / "empty.json"
    args = MODULE.parse_args(["--output-path", str(output), "--season-start", "2", "--season-end", "2"])

    class Scraper:
        async def get_tournaments_in_season(self, season):
            return []

    assert asyncio.run(MODULE.collect(args, Scraper(), {})) == 0
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["completeness"]["discovery_complete"] is True
    assert manifest["counts"]["discovery_failures"] == 0


def test_resume_excludes_source_confirmed_unplayed_row_without_losing_its_provenance(tmp_path):
    output = tmp_path / "history.json"
    argv = ["--output-path", str(output), "--tournament", "Fixture"]

    class Scraper:
        corrected = False
        failed_rows = []
        unplayed_rows = []
        calls = []

        async def get_matches_in_tournament(self, tournament):
            row = {"match_id": "2", "tournament_name": tournament, "url": "https://gol.gg/fixture",
                   "html": "<a href='2/page-preview/'>A vs B</a>", "error": "empty result"}
            self.failed_rows = [] if self.corrected else [row]
            self.unplayed_rows = [row] if self.corrected else []
            return [header()]

        async def get_games_in_match(self, mid):
            self.calls.append(mid)
            if mid == "2":
                raise ValueError("no completed maps")
            return [game()]

    scraper = Scraper()
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv), scraper, {})) == 2
    scraper.corrected = True
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv + ["--resume"]), scraper, {})) == 0
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv + ["--resume"]), scraper, {})) == 0
    assert {row["match_id"] for row in json.loads(output.read_text())} == {"1"}
    assert json.loads(output.with_suffix(".quarantine.json").read_text()) == []
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["counts"]["unplayed_matches"] == 1
    assert manifest["unplayed"]["2"]["tournament_name"] == "Fixture"


@pytest.mark.parametrize("stage", ["season", "tournament", "maps"])
def test_source_outage_interrupts_without_consuming_remaining_work(tmp_path, stage):
    output = tmp_path / "history.json"
    args = MODULE.parse_args([
        "--output-path", str(output), "--season-start", "2", "--season-end", "3",
        "--concurrency", "1", "--tournament-concurrency", "1",
    ])
    calls = []

    class Scraper:
        async def get_tournaments_in_season(self, season):
            calls.append(("season", season))
            if stage == "season":
                raise httpx.ConnectTimeout("source unreachable")
            return [{"trname": "First"}, {"trname": "Second"}]

        async def get_matches_in_tournament(self, tournament):
            calls.append(("tournament", tournament))
            if stage == "tournament":
                response = httpx.Response(429, request=httpx.Request("GET", "https://gol.gg/"))
                response.raise_for_status()
            return [header(), {**header(), "match_id": "2"}]

        async def get_games_in_match(self, mid):
            calls.append(("maps", mid))
            httpx.Response(429, request=httpx.Request("GET", "https://gol.gg/")).raise_for_status()

    with pytest.raises(httpx.HTTPError):
        asyncio.run(MODULE.collect(args, Scraper(), {}))
    assert sum(kind == stage for kind, _ in calls) == 1
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["counts"]["unresolved_matches"] == 0
    if stage == "maps":
        assert manifest["counts"]["pending_matches"] == 1
        assert manifest["counts"]["errored_matches"] == 1


def test_concurrent_requests_observe_global_minimum_interval(monkeypatch):
    clock = [0.0]
    sent_at = []

    async def sleep(delay):
        clock[0] += delay

    monkeypatch.setattr(MODULE.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(MODULE.asyncio, "sleep", sleep)

    async def run():
        def respond(request):
            sent_at.append(clock[0])
            return httpx.Response(200)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
            event_hooks={"request": [MODULE.RequestPacer(1.0)]},
        ) as client:
            await asyncio.gather(*(client.get("https://gol.gg/") for _ in range(3)))

    asyncio.run(run())
    assert sent_at == [0.0, 1.0, 2.0]


def test_retry_rounds_shrink_and_resume_skips_clean_and_quarantined_series(tmp_path, monkeypatch):
    output = tmp_path / "history.json"
    argv = ["--output-path", str(output), "--tournament", "Fixture", "--concurrency", "1",
            "--refresh-matches", "--refetch-games"]
    calls, sleeps = [], []

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(MODULE.asyncio, "sleep", sleep)

    class Scraper:
        recovered = False

        async def get_matches_in_tournament(self, tournament):
            return [{**header(), "match_id": str(mid)} for mid in range(1, 5)]

        async def get_games_in_match(self, mid):
            calls.append(mid)
            if mid == "3":
                raise ValueError("contradictory team identity")
            if (mid == "2" and calls.count(mid) < 3) or (mid == "4" and not self.recovered):
                raise httpx.RemoteProtocolError(
                    "server disconnected", request=httpx.Request("GET", f"https://gol.gg/game/stats/{mid}/page-summary/"),
                )
            return [{**game(mid), "match_id": mid}]

    scraper = Scraper()
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv), scraper, {})) == 2
    assert calls == ["1", "2", "3", "4", "2", "4", "2", "4", "4"]
    assert sleeps == [2, 5, 10]
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["counts"]["errored_matches"] == 1
    assert manifest["counts"]["unresolved_matches"] == 1
    assert manifest["counts"]["pending_matches"] == 0
    assert manifest["completeness"]["all_discovered_series_clean"] is False
    errors = json.loads(output.with_suffix(".errored.json").read_text())
    assert [row["match_id"] for row in errors] == ["4"]
    assert errors[0]["attempts"] == 4
    assert any("/4/page-summary/" in reason for reason in errors[0]["reasons"])
    assert [row["match_id"] for row in json.loads(output.with_suffix(".quarantine.json").read_text())] == ["3"]
    clean_before = json.loads(output.read_text())
    assert {row["match_id"] for row in clean_before} == {"1", "2"}
    scraper.recovered = True
    calls.clear()
    sleeps.clear()
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv + ["--resume"]), scraper, {})) == 2
    assert calls == ["4"]
    assert sleeps == []
    assert json.loads(output.with_suffix(".errored.json").read_text()) == []
    assert json.loads(output.read_text())[:2] == clean_before
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv + ["--resume"]), scraper, {})) == 2
    assert calls == ["4"]


def test_interrupted_error_queue_resumes_without_refetching_clean_series(tmp_path):
    output = tmp_path / "history.json"
    argv = ["--output-path", str(output), "--tournament", "Fixture", "--concurrency", "1",
            "--refresh-matches", "--refetch-games"]
    calls = []

    class Scraper:
        recovered = False

        async def get_matches_in_tournament(self, tournament):
            return [{**header(), "match_id": str(mid)} for mid in range(1, 4)]

        async def get_games_in_match(self, mid):
            calls.append(mid)
            if not self.recovered:
                if mid == "2":
                    raise httpx.ReadTimeout("read stalled")
                if mid == "3":
                    raise asyncio.CancelledError()
            return [{**game(mid), "match_id": mid}]

    scraper = Scraper()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(MODULE.collect(MODULE.parse_args(argv), scraper, {}))
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert {key: manifest["counts"][key] for key in
            ("clean_matches", "errored_matches", "pending_matches")} == {
                "clean_matches": 1, "errored_matches": 1, "pending_matches": 1,
            }
    scraper.recovered = True
    calls.clear()
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv + ["--resume"]), scraper, {})) == 0
    assert calls == ["2", "3"]
    clean = json.loads(output.read_text())
    calls.clear()
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv + ["--resume"]), scraper, {})) == 0
    assert calls == []
    assert json.loads(output.read_text()) == clean


def test_sustained_transport_outage_stops_without_exhausting_the_history(tmp_path):
    output = tmp_path / "history.json"
    args = MODULE.parse_args(["--output-path", str(output), "--tournament", "Fixture", "--concurrency", "1"])
    calls = []

    class Scraper:
        async def get_matches_in_tournament(self, tournament):
            return [{**header(), "match_id": str(mid)} for mid in range(25)]

        async def get_games_in_match(self, mid):
            calls.append(mid)
            raise httpx.RemoteProtocolError("server disconnected")

    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(MODULE.collect(args, Scraper(), {}))
    assert calls == [str(mid) for mid in range(20)]
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["counts"]["errored_matches"] == 20
    assert manifest["counts"]["pending_matches"] == 5
    assert json.loads(output.with_suffix(".quarantine.json").read_text()) == []


def test_failed_discovery_resume_preserves_unchanged_quarantine_evidence(tmp_path):
    output = tmp_path / "history.json"
    argv = ["--output-path", str(output), "--tournament", "Fixture"]
    calls = []

    class Scraper:
        failed_rows = [{"tournament_name": "Fixture", "error": "another row has no match ID", "html": "<tr/>"}]

        async def get_matches_in_tournament(self, tournament):
            return [header((2, 0))]

        async def get_games_in_match(self, mid):
            calls.append(mid)
            return [game()]

    scraper = Scraper()
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv), scraper, {})) == 2
    quarantine = json.loads(output.with_suffix(".quarantine.json").read_text())
    assert quarantine[0]["match"]["games"][0]["game_id"] == "1"
    assert json.loads(output.with_suffix(".manifest.json").read_text())["counts"]["discovery_failures"] == 1
    scraper.failed_rows = []
    assert asyncio.run(MODULE.collect(MODULE.parse_args(argv + ["--resume"]), scraper, {})) == 2
    assert calls == ["1"]
    assert json.loads(output.with_suffix(".quarantine.json").read_text()) == quarantine
