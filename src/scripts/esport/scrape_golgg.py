"""Fetch a separately named, identity-reconciled historical GOL.gg dataset."""
import argparse
import asyncio
import gzip
import hashlib
import json
import os
import resource
import sys
import time
from contextlib import aclosing
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from utils.scrapers.golgg import GolggScraper

ROLES = {"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"}
SCHEMA_VERSION = 1
RETRY_DELAYS = (2, 5, 10)
MAX_CONSECUTIVE_FETCH_ERRORS = 20
RETRYABLE_FETCH_ERRORS = (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value):
    return " ".join(str(value or "").casefold().split())


def score(value):
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(f"invalid nonnegative score: {value!r}")
    return int(value)


def reconcile_match(match, games, source_corrections=None):
    """Validate every map, then orient series to first-map IDs; never reorder maps."""
    raw = {key: value for key, value in match.get("source_header", match).items()
           if key not in {"games", "reconciliation", "source_header"}}
    if not games:
        raise ValueError("no complete maps returned")
    formats = {score(game.get("source_best_of")) for game in games}
    if len(formats) != 1 or 0 in formats:
        raise ValueError("missing or inconsistent source series format")
    best_of = formats.pop()
    mid = str(raw.get("match_id", ""))
    team_ids = (str(games[0].get("t1_id") or ""), str(games[0].get("t2_id") or ""))
    if not all(team_ids) or team_ids[0] == team_ids[1]:
        raise ValueError("map teams must have distinct nonempty IDs")
    wins = dict.fromkeys(team_ids, 0)
    game_ids = []
    for game in games:
        if best_of % 2 and max(wins.values()) >= best_of // 2 + 1:
            raise ValueError("extra map after completed series")
        gid = str(game.get("game_id") or "")
        if not gid or gid in game_ids:
            raise ValueError("missing or duplicate game ID")
        game_ids.append(gid)
        if str(game.get("match_id")) != mid:
            raise ValueError(f"map {gid} belongs to a different match")
        ids = (str(game.get("t1_id") or ""), str(game.get("t2_id") or ""))
        if ids[0] == ids[1] or set(ids) != set(team_ids):
            raise ValueError(f"map {gid} has contradictory team identities")
        if (game.get("t1_side"), game.get("t2_side")) != ("blue", "red"):
            raise ValueError(f"map {gid} does not preserve actual blue/red side order")
        flags = (game.get("t1_win"), game.get("t2_win"))
        if any(type(flag) is not bool for flag in flags) or sum(flags) != 1 or game.get("draw") is not False:
            raise ValueError(f"map {gid} requires exactly one boolean winner")
        player_ids = []
        for side in ("t1", "t2"):
            roster = game.get(f"{side}_players") or {}
            if set(roster) != ROLES:
                raise ValueError(f"map {gid} has incomplete {side} roster")
            side_ids = [str(player.get("player_id") or "") for player in roster.values()]
            if not all(side_ids) or len(set(side_ids)) != 5:
                raise ValueError(f"map {gid} has missing/duplicate {side} player IDs")
            player_ids.extend(side_ids)
        if len(set(player_ids)) != 10:
            raise ValueError(f"map {gid} opposing rosters overlap")
        wins[ids[0] if flags[0] else ids[1]] += 1
    for game in games:
        if "source_game_ids" in game and [str(gid) for gid in game["source_game_ids"]] != game_ids:
            raise ValueError("returned maps differ from complete source game IDs/order")
    if best_of % 2:
        if max(wins.values()) != best_of // 2 + 1:
            raise ValueError(f"incomplete BO{best_of} series")
    elif len(games) != best_of:
        raise ValueError(f"incomplete BO{best_of} fixed-map series")

    header_scores = (score(raw.get("t1_score")), score(raw.get("t2_score")))
    correction = (source_corrections or {}).get("series", {}).get(mid)
    names = (games[0].get("t1_name"), games[0].get("t2_name"))
    if not all(normalize_name(name) for name in names):
        raise ValueError("map team names are missing")
    if correction is not None:
        expected = {"t1_name": raw.get("sname_t1"), "t2_name": raw.get("sname_t2"),
                    "t1_score": header_scores[0], "t2_score": header_scores[1]}
        if correction.get("expected") != expected:
            raise ValueError("series source correction expected snapshot mismatch")
        if not isinstance(correction.get("provenance"), dict) or not correction["provenance"]:
            raise ValueError("series source correction lacks evidence provenance")
        corrected = correction.get("corrected", {})
        corrected_ids = (str(corrected.get("t1_id") or ""), str(corrected.get("t2_id") or ""))
        if corrected_ids[0] == corrected_ids[1] or set(corrected_ids) != set(team_ids):
            raise ValueError("series correction team IDs contradict map identities")
        corrected_scores = (score(corrected.get("t1_score")), score(corrected.get("t2_score")))
        if tuple(wins[tid] for tid in corrected_ids) != corrected_scores:
            raise ValueError("series correction score contradicts complete maps")
        orientation = "verified_source_correction"
    else:
        if sum(header_scores) != len(games):
            raise ValueError(f"header score total {sum(header_scores)} differs from {len(games)} complete maps")
        header_names = (normalize_name(raw.get("sname_t1")), normalize_name(raw.get("sname_t2")))
        map_names = tuple(normalize_name(name) for name in names)
        header_ids = (str(raw.get("t1_id") or ""), str(raw.get("t2_id") or ""))
        # Exact IDs take precedence. Compact link aliases are not identity evidence.
        if raw.get("score_alignment") == "unresolved":
            oriented_ids = None
            orientation = "ambiguous_aliases"
        elif all(header_ids):
            if header_ids[0] == header_ids[1] or set(header_ids) != set(team_ids):
                raise ValueError("header team IDs contradict map identities")
            oriented_ids = header_ids
            orientation = "exact_ids"
        elif header_names == map_names and map_names[0] != map_names[1]:
            oriented_ids = team_ids
            orientation = "exact_names"
        elif header_names == map_names[::-1] and map_names[0] != map_names[1]:
            oriented_ids = team_ids[::-1]
            orientation = "reversed_exact_names"
        else:
            oriented_ids = None
            orientation = "ambiguous_aliases"
        if oriented_ids is not None and tuple(wins[tid] for tid in oriented_ids) != header_scores:
            raise ValueError("identity-aligned header score contradicts map wins")
        if oriented_ids is None and sorted(header_scores) != sorted(wins.values()):
            raise ValueError("ambiguous header score distribution contradicts map wins")

    result = dict(raw)
    result.update({"source_header": raw, "t1_id": team_ids[0], "t2_id": team_ids[1],
                   "sname_t1": names[0], "sname_t2": names[1],
                   "t1_name": names[0], "t2_name": names[1],
                   "t1_score": wins[team_ids[0]], "t2_score": wins[team_ids[1]],
                   "games_played": len(games), "games": games,
                   "t1_win": wins[team_ids[0]] > wins[team_ids[1]],
                   "t2_win": wins[team_ids[1]] > wins[team_ids[0]],
                   "draw": wins[team_ids[0]] == wins[team_ids[1]],
                   "best_of": best_of,
                   "reconciliation": {"header_orientation": orientation, "complete_game_ids": game_ids,
                                      "source_correction": correction}})
    for game in games:
        for key in ("patch", "date", "tournament_name"):
            game[key] = raw.get(key)
        for key in ("t1_player_ids", "t2_player_ids", "t1_player_names", "t2_player_names", "t1_champions", "t2_champions"):
            game.pop(key, None)
    return result


async def bounded_results(items, operation, concurrency):
    """Keep only a fixed worker window alive, including tournament discovery."""
    iterator = iter(items)
    pending = {}
    try:
        while True:
            while len(pending) < concurrency:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending[asyncio.create_task(operation(item))] = item
            if not pending:
                break
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                item = pending.pop(task)
                try:
                    result = task.result()
                except Exception as error:
                    yield item, None, error
                else:
                    yield item, result, None
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def atomic_json(path, value=None, rows=None):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if rows is None:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            handle.write("[")
            for index, row in enumerate(rows):
                if index:
                    handle.write(",\n")
                json.dump(row, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("]\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class CollectionStore:
    """Append-only journal is authoritative; JSON outputs are atomic derived views."""
    def __init__(self, output_path, resume, config):
        self.output_path = output_path.resolve()
        self.manifest_path = self.output_path.with_suffix(".manifest.json")
        self.quarantine_path = self.output_path.with_suffix(".quarantine.json")
        self.errored_path = self.output_path.with_suffix(".errored.json")
        self.journal_path = self.output_path.with_suffix(".journal.jsonl")
        self.records = {}
        self.unplayed = {}
        self.discovery = {}
        self.config = config
        self.created_at = timestamp()
        self.generation = 1
        paths = (self.output_path, self.manifest_path, self.quarantine_path, self.errored_path, self.journal_path)
        if any(path.exists() for path in paths):
            if not resume or not self.journal_path.exists():
                raise ValueError("refusing existing/unowned output; choose a new --output-path or --resume an owned journal")
            self._replay()
        else:
            if resume:
                raise ValueError("--resume requires an existing owned collection journal")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._append({"op": "init", "schema_version": SCHEMA_VERSION, "config": config,
                          "created_at": self.created_at, "output_path": str(self.output_path)})

    def _replay(self):
        with self.journal_path.open("r+b") as handle:
            first = True
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.truncate(offset)  # Only an interrupted final append is discardable.
                    break
                event = json.loads(line)
                if first:
                    if (event.get("op") != "init" or event.get("schema_version") != SCHEMA_VERSION
                            or event.get("config") != self.config or event.get("output_path") != str(self.output_path)):
                        raise ValueError("resume journal does not match output/config/source corrections; use a new output path")
                    self.created_at = event["created_at"]
                    first = False
                elif event["op"] == "match":
                    self.records[event["record"]["match_id"]] = event["record"]
                    self.unplayed.pop(event["record"]["match_id"], None)
                elif event["op"] == "unplayed":
                    self.unplayed[event["match_id"]] = event["source"]
                    self.records.pop(event["match_id"], None)
                elif event["op"] == "discovery":
                    self.discovery[event["key"]] = event["value"]
                elif event["op"] == "generation":
                    self.generation = event["generation"]
                    self.discovery.clear()
            if first:
                raise ValueError("collection journal has no complete initialization record")

    def _append(self, event):
        with self.journal_path.open("a", encoding="utf-8") as handle:
            json.dump(event, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def put(self, match, status, reasons):
        mid = str(match.get("match_id") or "")
        if not mid:
            raise ValueError("cannot persist a discovered match without its ID")
        old = self.records.get(mid, {})
        record = {"match_id": mid, "status": status, "match": match, "reasons": reasons,
                  "attempts": old.get("attempts", 0) + (status in {"clean", "unresolved", "errored"}),
                  "generation": self.generation, "updated_at": timestamp()}
        self._append({"op": "match", "record": record})
        self.records[mid] = record
        self.unplayed.pop(mid, None)

    def exclude_unplayed(self, source):
        mid = str(source.get("match_id") or "")
        if not mid:
            raise ValueError("cannot account an unplayed source row without its ID")
        if self.unplayed.get(mid) == source:
            return
        self._append({"op": "unplayed", "match_id": mid, "source": source})
        self.unplayed[mid] = source
        self.records.pop(mid, None)

    def discovery_result(self, key, status, **metadata):
        value = {"status": status, "updated_at": timestamp(), **metadata}
        self._append({"op": "discovery", "key": key, "value": value})
        self.discovery[key] = value

    def new_generation(self):
        self.generation += 1
        self._append({"op": "generation", "generation": self.generation})
        self.discovery.clear()

    def checkpoint(self, status):
        counts = {"discovered_matches": len(self.records), "clean_matches": 0,
                  "unresolved_matches": 0, "errored_matches": 0, "pending_matches": 0, "clean_games": 0,
                  "unplayed_matches": len(self.unplayed),
                  "discovery_failures": sum(item["status"] != "complete" for item in self.discovery.values()),
                  "malformed_source_rows": sum(len(item.get("malformed_rows", [])) for item in self.discovery.values())}
        for record in self.records.values():
            counts[record["status"] + "_matches"] += 1
            if record["status"] == "clean":
                counts["clean_games"] += len(record["match"]["games"])
        atomic_json(self.output_path, rows=(record["match"] for record in self.records.values() if record["status"] == "clean"))
        atomic_json(self.quarantine_path, rows=(record for record in self.records.values() if record["status"] == "unresolved"))
        atomic_json(self.errored_path, rows=(record for record in self.records.values() if record["status"] == "errored"))
        manifest = {"schema_version": SCHEMA_VERSION, "status": status, "created_at": self.created_at,
                    "updated_at": timestamp(), "generation": self.generation, "config": self.config,
                    "counts": counts, "discovery": self.discovery,
                    "unplayed": self.unplayed,
                    "output_path": str(self.output_path), "quarantine_path": str(self.quarantine_path),
                    "errored_path": str(self.errored_path),
                    "fetch_policy": {"retry_delays_seconds": RETRY_DELAYS,
                                     "max_consecutive_errors": MAX_CONSECUTIVE_FETCH_ERRORS},
                    "journal_path": str(self.journal_path), "journal_authoritative": True,
                    "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                    "completeness": {"all_discovered_ids_accounted": True,
                                     "all_discovered_series_clean": counts["clean_matches"] == counts["discovered_matches"],
                                     "discovery_complete": status in {"complete", "incomplete"} and not counts["discovery_failures"],
                                     "point_in_time_training_evidence": False}}
        atomic_json(self.manifest_path, manifest)
        print(f"[{status}] discovered={counts['discovered_matches']} clean={counts['clean_matches']} "
              f"pending={counts['pending_matches']} errored={counts['errored_matches']} unresolved={counts['unresolved_matches']} "
              f"discovery_failures={counts['discovery_failures']} games={counts['clean_games']} "
              f"peak_rss_mib={manifest['peak_rss_mib']}", flush=True)
        return manifest


class RequestPacer:
    """Space request starts globally, including concurrent map/stat fetches."""

    def __init__(self, interval):
        self.interval = interval
        self.lock = asyncio.Lock()
        self.next_at = 0.0

    async def __call__(self, request):
        async with self.lock:
            delay = self.next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_at = time.monotonic() + self.interval


def raise_for_source_outage(error):
    """Do not turn an unavailable source into thousands of invalid records."""
    if isinstance(error, httpx.TransportError) or (
        isinstance(error, httpx.HTTPStatusError)
        and (error.response.status_code in {403, 429} or error.response.status_code >= 500)
    ):
        raise error


class SourceArchive:
    def __init__(self, path):
        self.path = path.resolve()
        self.path.mkdir(parents=True, exist_ok=True)

    async def response(self, response):
        body = await response.aread()
        digest = hashlib.sha256(body).hexdigest()
        destination = self.path / digest[:2] / f"{digest}.html.gz"
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp")
            with gzip.open(temporary, "wb") as handle:
                handle.write(body)
            temporary.replace(destination)
        request = response.request
        metadata = {"retrieved_at": timestamp(), "url": str(request.url), "method": request.method,
                    "request_body": request.content.decode("utf-8", errors="replace"),
                    "status_code": response.status_code, "sha256": digest,
                    "body_path": str(destination.relative_to(self.path)),
                    "content_type": response.headers.get("content-type")}
        with (self.path / "requests.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True, help="New absolute JSON path; never an existing dataset")
    parser.add_argument("--resume", action="store_true", help="Recover this collector's owned journal")
    parser.add_argument("--new-generation", action="store_true", help="Explicitly invalidate prior maps/discovery when resuming with refresh/refetch")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum active series")
    parser.add_argument("--map-concurrency", "--max-pages", dest="max_pages", type=int, default=4, help="Maximum parser HTTP requests")
    parser.add_argument("--tournament-concurrency", type=int, default=2)
    parser.add_argument("--request-interval", type=float, default=1.0, help="Minimum seconds between HTTP request starts")
    parser.add_argument("--batch-size", type=int, default=500, help="Snapshot interval; journal commits every result")
    parser.add_argument("--refresh-matches", action="store_true")
    parser.add_argument("--refetch-games", action="store_true")
    parser.add_argument("--match-id")
    parser.add_argument("--tournament", action="append", default=[], help="Exact tournament name; repeatable")
    parser.add_argument("--match-source", type=Path, help="JSON header object/list for targeted --match-id; no global discovery")
    parser.add_argument("--season-start", type=int, default=2)
    parser.add_argument("--season-end", type=int, default=16)
    parser.add_argument("--source-corrections", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.output_path.is_absolute() or args.output_path.suffix != ".json":
        parser.error("--output-path must be a new absolute .json path")
    if min(args.concurrency, args.max_pages, args.tournament_concurrency, args.batch_size) < 1:
        parser.error("concurrency and batch-size values must be positive")
    if args.season_start > args.season_end:
        parser.error("season-start must not exceed season-end")
    if not 0 < args.request_interval < float("inf"):
        parser.error("request-interval must be finite and positive")
    if args.match_source and not args.match_id:
        parser.error("--match-source requires --match-id")
    if args.match_id and not (args.match_source or args.tournament or args.resume):
        parser.error("--match-id requires --tournament, --match-source, or an owned --resume journal")
    if args.new_generation and not (args.resume and (args.refresh_matches or args.refetch_games)):
        parser.error("--new-generation requires --resume and --refresh-matches or --refetch-games")
    return args


async def collect(args, scraper, corrections):
    config = {"match_id": args.match_id, "tournaments": sorted(args.tournament),
              "season_start": args.season_start, "season_end": args.season_end,
              "match_source": str(args.match_source.resolve()) if args.match_source else None,
              "match_source_sha256": hashlib.sha256(args.match_source.read_bytes()).hexdigest() if args.match_source else None,
              "source_corrections_sha256": hashlib.sha256(json.dumps(corrections, sort_keys=True).encode()).hexdigest(),
              "archive_dir": str(args.archive_dir.resolve()) if args.archive_dir else None,
              "refresh_matches": args.refresh_matches, "refetch_games": args.refetch_games}
    store = CollectionStore(args.output_path, args.resume, config)
    # Resume never invalidates reconciled maps, even after a completed collection.
    if args.new_generation:
        store.new_generation()
    target = str(args.match_id) if args.match_id else None
    for mid, record in list(store.records.items()):
        if (not target or target == mid) and (args.refetch_games or args.refresh_matches) and record["generation"] != store.generation:
            raw = dict(record["match"].get("source_header", record["match"]))
            raw.pop("games", None)
            store.put(raw, "pending", ["new refresh/refetch generation; stale maps invalidated"])
    store.checkpoint("running")
    print("Resource plan: one in-memory dataset; streaming compact snapshots; append-only per-record journal; "
          "compressed source bodies. Journal is authoritative after interruption. No original dataset is loaded.", flush=True)

    def ingest(rows):
        for row in rows:
            mid = str(row.get("match_id") or "")
            if not mid:
                raise ValueError("discovery returned row without match_id")
            if target and mid != target:
                continue
            previous = store.records.get(mid)
            raw = dict(row.get("source_header", row))
            raw.pop("games", None)
            if previous:
                old_header = dict(previous["match"].get("source_header", previous["match"]))
                old_header.pop("games", None)
                if old_header == raw:
                    continue
            store.put(raw, "pending", ["awaiting complete fresh maps"])

    try:
        if args.match_source:
            key = "match_source"
            if store.discovery.get(key, {}).get("status") != "complete":
                try:
                    rows = json.loads(args.match_source.read_text())
                    ingest(rows if isinstance(rows, list) else [rows])
                    store.discovery_result(key, "complete")
                except Exception as error:
                    store.discovery_result(key, "failed", error=f"{type(error).__name__}: {error}")
        tournaments = {name: {"trname": name} for name in args.tournament}
        if not target and not args.tournament:
            # Sequential season discovery avoids unbounded season fanout and journals each response.
            for season in range(args.season_start, args.season_end + 1):
                key = f"season:{season}"
                cached = store.discovery.get(key, {})
                if cached.get("status") == "complete":
                    rows = cached["tournaments"]
                else:
                    try:
                        rows = await scraper.get_tournaments_in_season(season)
                        if not isinstance(rows, list):
                            raise ValueError("season discovery returned a non-list response")
                        if any(not row.get("trname") for row in rows):
                            raise ValueError("season discovery returned unnamed tournament")
                        store.discovery_result(key, "complete", tournaments=rows)
                    except Exception as error:
                        store.discovery_result(key, "failed", error=f"{type(error).__name__}: {error}")
                        raise_for_source_outage(error)
                        continue
                tournaments.update({row["trname"]: row for row in rows})
        todo = [row for name, row in tournaments.items() if store.discovery.get(f"tournament:{name}", {}).get("status") != "complete"]

        async def discover(tournament):
            return await scraper.get_matches_in_tournament(tournament["trname"])

        async for tournament, rows, error in bounded_results(todo, discover, args.tournament_concurrency):
            name = tournament["trname"]
            key = f"tournament:{name}"
            if error is not None:
                store.discovery_result(key, "failed", error=f"{type(error).__name__}: {error}")
                raise_for_source_outage(error)
            else:
                try:
                    ingest(rows)
                    for row in getattr(scraper, "unplayed_rows", []):
                        if row.get("tournament_name") == name and (
                            not target or str(row.get("match_id")) == target
                        ):
                            store.exclude_unplayed(row)
                    failures = [
                        row for row in getattr(scraper, "failed_rows", [])
                        if row.get("tournament_name", row.get("tournament")) == name
                        and (not target or str(row.get("match_id")) == target)
                    ]
                    if failures:
                        # Keep malformed source rows, including any recoverable IDs, in the journal/manifest.
                        for failure in failures:
                            if failure.get("match_id") and (not target or str(failure["match_id"]) == target):
                                store.put(failure, "unresolved", [failure.get("error", "malformed source row")])
                        store.discovery_result(key, "failed", malformed_rows=failures, rows=len(rows))
                    elif not rows and score(tournament.get("nbgames", 0)) > 0:
                        store.discovery_result(key, "failed", error="nonempty tournament returned no matches")
                    else:
                        store.discovery_result(key, "complete", rows=len(rows), expected_games=tournament.get("nbgames"))
                except Exception as failure:
                    store.discovery_result(key, "failed", error=f"{type(failure).__name__}: {failure}")
            print(f"Discovery {key}: {store.discovery[key]['status']}; IDs={len(store.records)}", flush=True)
        if target and target not in store.records:
            store.put({"match_id": target}, "unresolved", ["target not found in supplied source/tournament"])
        store.checkpoint("running")
        pending = [mid for mid, record in store.records.items()
                   if record["status"] in {"pending", "errored"} and (not target or target == mid)]

        async def fetch(mid):
            raw = dict(store.records[mid]["match"].get("source_header", store.records[mid]["match"]))
            raw.pop("games", None)
            games = await scraper.get_games_in_match(mid)
            try:
                return reconcile_match(raw, games, corrections), []
            except ValueError as error:
                raw["games"] = games
                return raw, [f"{type(error).__name__}: {error}"]

        consecutive_errors = 0
        for round_index, delay in enumerate((0, *RETRY_DELAYS), start=1):
            if not pending:
                break
            if delay:
                print(f"Retry round {round_index}: {len(pending)} errored series after {delay}s rest", flush=True)
                await asyncio.sleep(delay)
            completed = 0
            async with aclosing(bounded_results(pending, fetch, args.concurrency)) as results:
                async for mid, row, error in results:
                    if error is not None:
                        raw = dict(store.records[mid]["match"].get("source_header", store.records[mid]["match"]))
                        raw.pop("games", None)
                        reasons = [f"{type(error).__name__}: {error}"]
                        if isinstance(error, httpx.HTTPError):
                            try:
                                request = error.request
                            except RuntimeError:
                                pass  # Manually raised HTTP errors can lack request metadata.
                            else:
                                reasons.append(f"{request.method} {request.url}")
                            store.put(raw, "errored", reasons)
                            if not isinstance(error, RETRYABLE_FETCH_ERRORS):
                                raise_for_source_outage(error)
                            consecutive_errors += 1
                            if consecutive_errors >= MAX_CONSECUTIVE_FETCH_ERRORS:
                                raise error
                        else:
                            consecutive_errors = 0
                            store.put(raw, "unresolved", reasons)
                    else:
                        consecutive_errors = 0
                        match, reasons = row
                        store.put(match, "unresolved" if reasons else "clean", reasons)
                    completed += 1
                    reasons = store.records[mid]["reasons"]
                    print(f"Maps round={round_index} {completed}/{len(pending)} match={mid} status={store.records[mid]['status']}"
                          + (f" reasons={'; '.join(reasons)}" if reasons else ""), flush=True)
                    if completed % args.batch_size == 0:
                        store.checkpoint("running")
            pending = [mid for mid in pending if store.records[mid]["status"] == "errored"]
            if pending and round_index <= len(RETRY_DELAYS):
                store.checkpoint("running")
        incomplete = any(record["status"] != "clean" for record in store.records.values()) or any(item["status"] != "complete" for item in store.discovery.values())
        store.checkpoint("incomplete" if incomplete else "complete")
        return 2 if incomplete else 0
    except BaseException:
        store.checkpoint("interrupted")
        raise


async def main(argv=None):
    args = parse_args(argv)
    corrections = json.loads(args.source_corrections.read_text()) if args.source_corrections else {}
    if not isinstance(corrections, dict):
        raise ValueError("source corrections must be a JSON object")
    async with GolggScraper(max_pages=args.max_pages, source_corrections=corrections) as scraper:
        scraper.client.event_hooks.setdefault("request", []).append(RequestPacer(args.request_interval))
        if args.archive_dir:
            scraper.client.event_hooks.setdefault("response", []).append(SourceArchive(args.archive_dir).response)
        return await collect(args, scraper, corrections)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
