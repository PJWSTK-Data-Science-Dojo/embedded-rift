import json
from typing import Self
import httpx
import re
from urllib.parse import urljoin, quote
import asyncio
import parsel

GOLGG_URL = "https://gol.gg"
GOLGG_TOURNAMENT_API = "https://gol.gg/tournament/ajax.trlist.php"
GOLGG_MATCH_SUMMARY = "https://gol.gg/game/stats/{}/page-summary/"
INDEX_TO_ROLE = {
    0: "TOP",
    1: "JUNGLE",
    2: "MID",
    3: "ADC",
    4: "SUPPORT",
}
GAME_FETCH_RETRIES = 3


async def _gather_owned(*coroutines):
    """Settle every child before returning a result, error, or cancellation."""
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def extract_champion_from_player_row(player_row: parsel.Selector) -> dict:
    """Extract champion metadata from a GOL.gg player row."""
    champion_links = player_row.css('a[href*="champion/champion-stats"]')
    if not champion_links:
        return {
            "champion_id": None,
            "champion_name": None,
            "champion_image": None,
        }

    champion_link = champion_links[0]
    champion_href = champion_link.css("::attr(href)").get()
    champion_img = champion_link.css("img")
    champion_name = champion_img.css("::attr(alt)").get()
    champion_image = champion_img.css("::attr(src)").get()

    champion_id = None
    match = re.search(r"champion-stats/(\d+)/", champion_href or "")
    if match:
        champion_id = match.group(1)

    return {
        "champion_id": champion_id,
        "champion_name": champion_name,
        "champion_image": urljoin(GOLGG_URL, champion_image) if champion_image else None,
    }


def infer_best_of(t1_score: int, t2_score: int) -> int | None:
    """Infer best-of from a completed match score."""
    if t1_score == t2_score:
        games_played = t1_score + t2_score
        return games_played if games_played > 0 else None

    wins_needed = max(t1_score, t2_score)
    if wins_needed <= 0:
        return None
    return (wins_needed * 2) - 1


def score_for_link_order(team_a, team_b, result_left_team, result_right_team, score_left, score_right):
    """Align score cells by identity, never by a presumed winner-first layout."""
    def normalized(name):
        return re.sub(r"[^a-z0-9]+", "", (name or "").lower())

    a, b, left, right = map(normalized, (team_a, team_b, result_left_team, result_right_team))
    direct = bool((left and left == a) or (right and right == b))
    reverse = bool((right and right == a) or (left and left == b))
    if direct == reverse:
        return score_left, score_right, "unresolved"
    if reverse:
        return score_right, score_left, "reversed"
    return score_left, score_right, "direct"


class GolggScraper:
    def __init__(self, max_pages: int = 20, *, source_corrections: dict | None = None):
        self.semaphore = asyncio.Semaphore(max_pages)
        self.client: httpx.AsyncClient | None = None
        self.source_corrections = source_corrections or {}
        self.failed_rows = []
        self.unplayed_rows = []

    async def start(self, headless: bool = True) -> Self:
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
            },
            follow_redirects=True,
            timeout=30.0,
            limits=httpx.Limits(
                max_connections=max(100, self.semaphore._value * 4),
                max_keepalive_connections=max(20, self.semaphore._value),
            ),
        )
        return self

    async def stop(self):
        if self.client:
            await self.client.aclose()

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def get_tournaments_in_season(self, season: int = 9) -> list[dict]:
        async with self.semaphore:
            response = await self.client.post(
                GOLGG_TOURNAMENT_API,
                data={"season": f"S{season}"},
            )
        response.raise_for_status()
        data = response.json()
        return data

    async def get_all_tournaments(self) -> list[dict]:
        result = []
        for season in range(2, 17):
            data = await self.get_tournaments_in_season(season)
            result.extend(data)
        return result

    async def get_matches_in_tournament(self, tournament_name: str) -> list[dict]:
        """Process a single tournament using a dedicated page."""
        failed = []
        encoded_trname = quote(tournament_name)
        s = f"tournament-matchlist/{encoded_trname}/"

        url = f"{GOLGG_URL}/tournament/{s}"

        async with self.semaphore:
            response = await self.client.get(url)
        response.raise_for_status()
        html = response.content.decode("utf-8")

        sel = parsel.Selector(text=html)
        tables = sel.css(".table_list")
        matches_table = next(
            (table for table in tables if "data-sort" in table.attrib), None
        )

        if not matches_table:
            raise ValueError(f"Missing games table in {url}")

        # Extract links and match IDs
        rows = matches_table.css("tbody tr")
        matches = []

        for row in rows:
            match_id = None
            try:
                href = row.css("a::attr(href)").get()

                if not href:
                    continue
                link_text = row.css("a::text").get()
                if not link_text or " vs " not in link_text:
                    continue
                # Extract match ID using regex
                pattern = r"stats/(\d+)/"
                match = re.search(pattern, href)
                if not match:
                    print("Couldn't extract match id from", href)
                    continue

                match_id = match.group(1)
                team_a, team_b = link_text.split(" vs ")

                team_a = team_a.strip()
                team_b = team_b.strip()
                won = row.css("td.text_victory::text").get()
                lost = row.css("td.text_defeat::text").get()
                cells = row.css("td")
                result_left_team = cells[1].xpath("string(.)").get().strip()
                result_right_team = cells[3].xpath("string(.)").get().strip()
                score = row.css("td:nth-child(3)::text").get()
                if "/page-preview/" in href and not won and not lost and (
                    not score or not score.replace("-", "").strip()
                ):
                    self.unplayed_rows.append({
                        "match_id": match_id, "tournament_name": tournament_name,
                        "url": url, "html": row.get(),
                    })
                    continue
                if not score or "-" not in score:
                    raise ValueError("Missing completed match result")
                score = score.strip()

                score_left, score_right = score.split("-")
                score_left = int(score_left.strip())
                score_right = int(score_right.strip())

                team_a_score, team_b_score, score_alignment = score_for_link_order(
                    team_a, team_b, result_left_team, result_right_team, score_left, score_right
                )

                patch = row.css("td:nth-child(6)::text").get()
                date = row.css("td:nth-child(7)::text").get()
                data = {
                    "match_id": match_id,
                    "tournament_name": tournament_name,
                    "link": href,
                    "sname_t1": team_a,
                    "sname_t2": team_b,
                    "won": won,
                    "lost": lost,
                    "score": score,
                    "result_left_team": result_left_team,
                    "result_right_team": result_right_team,
                    "score_left": score_left,
                    "score_right": score_right,
                    "score_alignment": score_alignment,
                    "t1_score": team_a_score,
                    "t2_score": team_b_score,
                    "games_played": team_a_score + team_b_score,
                    "t1_win": team_a_score > team_b_score,
                    "t2_win": team_b_score > team_a_score,
                    "draw": team_a_score == team_b_score,
                    "best_of": infer_best_of(team_a_score, team_b_score),
                    "patch": patch,
                    "date": date,
                }
            except Exception as e:
                print("Error processing row:", e)
                failed.append(row.get())
                self.failed_rows.append({"match_id": match_id, "tournament_name": tournament_name, "url": url, "html": row.get(), "error": str(e)})
                continue
            matches.append(data)
        if failed:
            print(f"Skipped {len(failed)} malformed rows for tournament {tournament_name}")
        return matches

    async def get_game_selector(self, game_id: str) -> parsel.Selector:
        s = f"/game/stats/{game_id}/page-game/"
        url = f"{GOLGG_URL}{s}"

        async with self.semaphore:
            response = await self.client.get(url)
        response.raise_for_status()
        html = response.content.decode("utf-8")

        sel = parsel.Selector(text=html)
        return sel

    async def get_players_stats(self, game_id: str) -> dict[str, dict]:
        s = f"/game/stats/{game_id}/page-fullstats/"
        url = f"{GOLGG_URL}{s}"
        async with self.semaphore:
            response = await self.client.get(url)
        response.raise_for_status()
        html = response.content.decode("utf-8")
        sel = parsel.Selector(text=html)
        table = sel.css(".completestats")
        rows = table.xpath("./tr | ./tbody/tr")
        if not rows:
            raise ValueError(f"Game {game_id}: missing full player statistics")
        result = {
            "blue": {
                "TOP": {},
                "JUNGLE": {},
                "MID": {},
                "ADC": {},
                "SUPPORT": {},
            },
            "red": {
                "TOP": {},
                "JUNGLE": {},
                "MID": {},
                "ADC": {},
                "SUPPORT": {},
            },
        }

        for row in rows:
            tds = row.css("td::text").getall()
            if not tds:
                continue
            title, *stats = tds
            title = title.strip()
            title = title.replace(" ", "_").replace(":", "").replace("'", "").lower()
            if len(stats) != 10:
                stats = [None] * 10

            if title.endswith("%"):
                stats = [float(s[:-1]) / 100 if s else None for s in stats]
            else:
                stats = [
                    (
                        float(s)
                        if s and s.replace("-", "").replace(".", "").isnumeric()
                        else s
                    )
                    for s in stats
                ]

            blue_stats = stats[:5]
            red_stats = stats[5:]

            for i, (bs, rs) in enumerate(zip(blue_stats, red_stats)):
                role = INDEX_TO_ROLE.get(i, "UNKNOWN")

                result["blue"][role][title] = bs
                result["red"][role][title] = rs

        return result

    def _game_identity(self, game_sel: parsel.Selector, game_id: str | None = None) -> dict:
        """Resolve actual map sides, permitting only expectation-guarded evidence overrides."""
        raw = {}
        tables = game_sel.css(".playersInfosLine")
        if len(tables) != 2:
            raise ValueError(f"Game {game_id}: expected two side rosters")
        for side, table in zip(("blue", "red"), tables):
            header = game_sel.css(f".col-cadre .{side}-line-header")
            links = header.css('a[href*="teams/team-stats/"]')
            if len(links) != 1:
                raise ValueError(f"Game {game_id}: missing or ambiguous {side} identity")
            team_match = re.search(r"teams/team-stats/(\d+)/", links[0].attrib["href"])
            if not team_match:
                raise ValueError(f"Game {game_id}: invalid {side} team ID")
            raw[f"{side}_id"] = team_match.group(1)
            raw[f"{side}_name"] = links[0].xpath("string(.)").get().strip()
            # WIN/LOSS is a trailing label on the actual side header, not summary CSS.
            outcome = re.search(r"-\s*(WIN|LOSS)\s*$", header.xpath("string(.)").get().strip(), re.I)
            raw[f"{side}_win"] = outcome.group(1).upper() == "WIN" if outcome else None
            raw[f"{side}_player_ids"] = [
                re.search(r"player-stats/(\d+)/", href).group(1)
                for href in table.css('a[href*="players/player-stats/"]::attr(href)').getall()
            ]
        identity = dict(raw)
        correction = self.source_corrections.get("games", {}).get(str(game_id))
        if correction is not None:
            if not isinstance(correction.get("provenance"), dict) or not correction["provenance"]:
                raise ValueError(f"Game {game_id}: correction requires evidence provenance")
            expected = correction.get("expected", {})
            if set(expected) != set(raw):
                raise ValueError(f"Game {game_id}: correction requires a complete original identity snapshot")
            for key, actual in raw.items():
                wanted = expected[key]
                matches = sorted(wanted) == sorted(actual) if key.endswith("_player_ids") else wanted == actual
                if not matches:
                    raise ValueError(f"Game {game_id}: correction expectation mismatch for {key}")
            corrected = correction.get("corrected", {})
            allowed = {"blue_id", "red_id", "blue_name", "red_name", "blue_win", "red_win"}
            if not corrected or not set(corrected) <= allowed:
                raise ValueError(f"Game {game_id}: invalid correction fields")
            identity.update(corrected)
        if identity["blue_id"] == identity["red_id"]:
            raise ValueError(f"Game {game_id}: duplicate team IDs; distinct sides require verified source correction")
        for side in ("blue", "red"):
            if not isinstance(identity[f"{side}_id"], str) or not identity[f"{side}_id"].isdigit():
                raise ValueError(f"Game {game_id}: invalid corrected team ID")
            if not isinstance(identity[f"{side}_name"], str) or not identity[f"{side}_name"].strip():
                raise ValueError(f"Game {game_id}: missing team name")
            players = identity[f"{side}_player_ids"]
            if len(players) != 5 or len(set(players)) != 5:
                raise ValueError(f"Game {game_id}: {side} requires five distinct players")
        if set(identity["blue_player_ids"]) & set(identity["red_player_ids"]):
            raise ValueError(f"Game {game_id}: opposing rosters share player IDs")
        if any(type(identity[f"{side}_win"]) is not bool for side in ("blue", "red")):
            raise ValueError(f"Game {game_id}: missing side winner flags")
        if identity["blue_win"] == identity["red_win"]:
            raise ValueError(f"Game {game_id}: contradictory side winner flags")
        return {**identity, "raw": raw, "correction": correction}

    async def get_players_in_game(self, game_sel: parsel.Selector, *, game_id: str | None = None) -> dict[str, dict]:
        """Return separate, validated side rosters keyed by actual map team IDs."""
        identity = self._game_identity(game_sel, game_id)
        team_1_id, team_2_id = identity["blue_id"], identity["red_id"]
        t1_players_table, t2_players_table = game_sel.css(".playersInfosLine")

        def parse_players(rows: list[parsel.Selector]) -> dict[str, dict]:
            players = {}
            for i, player in enumerate(rows):
                player_td = player.css("td")[0]
                champion_data = extract_champion_from_player_row(player)
                player_link = player_td.css('a[href*="players/player-stats/"]')[0]
                href = player_link.css("::attr(href)").get()
                player_id_match = re.search(r"player-stats/(\d+)/", href or "")
                role = INDEX_TO_ROLE.get(i, "UNKNOWN")
                players[role] = {
                    "player_id": player_id_match.group(1) if player_id_match else None,
                    "player_name": player_link.css("::text").get(),
                    **champion_data,
                }
            return players

        return {
            team_1_id: parse_players(t1_players_table.xpath("./tr | ./tbody/tr")),
            team_2_id: parse_players(t2_players_table.xpath("./tr | ./tbody/tr")),
        }

    async def get_team_stats(self, game_sel: parsel.Selector, *, game_id: str | None = None) -> dict[str, dict]:
        result = {
            "blue_id": None,
            "red_id": None,
            "gameDuration": 0,
            "blue": {
                "kills": 0,
                "towers": 0,
                "dragons": 0,
                "nashors": 0,
                "gold": 0,
            },
            "red": {
                "kills": 0,
                "towers": 0,
                "dragons": 0,
                "nashors": 0,
                "gold": 0,
            },
        }

        team_info = game_sel.css(".col-cadre")[0]
        dur_row, stats_row = team_info.xpath("./div")
        dur_text = dur_row.css("h1::text").get().strip()
        duration = None
        if dur_text:
            minutes, seconds = dur_text.split(":")
            duration = int(minutes) * 60 + int(seconds)
        result["gameDuration"] = duration
        blue, red = stats_row.xpath("./*")
        bteam, bstats, champions = blue.xpath("./*")
        rteam, rstats, _ = red.xpath("./*")
        identity = self._game_identity(game_sel, game_id)
        result["blue_id"] = identity["blue_id"]
        result["red_id"] = identity["red_id"]

        kills, towers, dragons, nashors, gold, _ = bstats.xpath("./*")
        kills = kills.css("span::text").get().strip()
        towers = towers.css("span::text").get().strip()
        dragons = dragons.css("span::text").get()
        nashors = nashors.css("span::text").get()
        if not dragons:
            dragons = None
        else:
            dragons = int(dragons.strip())

        if not nashors:
            nashors = None
        else:
            nashors = int(nashors.strip())

        gold = gold.css("::text").get().strip()
        result["blue"]["kills"] = int(kills) if kills else 0
        result["blue"]["towers"] = int(towers) if towers else 0
        result["blue"]["dragons"] = int(dragons) if dragons else 0
        result["blue"]["nashors"] = nashors
        result["blue"]["gold"] = float(gold[:-1]) * 1000 if gold else 0
        kills, towers, dragons, nashors, gold, _ = rstats.xpath("./*")
        kills = kills.css("span::text").get().strip()
        towers = towers.css("span::text").get().strip()
        dragons = dragons.css("span::text").get()
        nashors = nashors.css("span::text").get()
        gold = gold.css("span::text").get().strip()
        if not dragons:
            dragons = None
        else:
            dragons = int(dragons.strip())

        if not nashors:
            nashors = None
        else:
            nashors = int(nashors.strip())

        result["red"]["kills"] = int(kills) if kills else 0
        result["red"]["towers"] = int(towers) if towers else 0
        result["red"]["dragons"] = int(dragons) if dragons else 0
        result["red"]["nashors"] = nashors
        result["red"]["gold"] = float(gold[:-1]) * 1000 if gold else 0

        return result

    async def get_games_in_match(self, match_id: str) -> list[dict]:
        """Return every listed map in actual blue/red order, or reject the series."""
        if not self.client:
            raise RuntimeError("GolggScraper is not started")
        url = f"{GOLGG_URL}/game/stats/{match_id}/page-summary/"
        async with self.semaphore:
            response = await self.client.get(url)
        response.raise_for_status()
        sel = parsel.Selector(text=response.content.decode("utf-8"))
        game_links = [
            link for link in sel.css("#gameMenuToggler li > a")
            if re.fullmatch(r"Game\s+\d+", link.xpath("string(.)").get().strip(), re.I)
        ]
        game_ids = []
        for link in game_links:
            found = re.search(r"game/stats/(\d+)/", link.attrib.get("href", ""))
            if not found:
                raise ValueError(f"Match {match_id}: invalid game link")
            game_ids.append(found.group(1))
        # BO1 summary URLs may redirect to the map page instead.
        if not game_ids and sel.css(".playersInfosLine"):
            game_ids = [str(match_id)]
        if not game_ids or len(set(game_ids)) != len(game_ids):
            raise ValueError(f"Match {match_id}: missing or duplicate game IDs")
        if not sel.css(".playersInfosLine"):
            tables = sel.css(".col-cadre")
            if not tables or len(tables[0].xpath("./*")) - 1 != len(game_ids):
                raise ValueError(f"Match {match_id}: summary map rows do not match game links")
        cadres = sel.css(".col-cadre")
        header_text = cadres[0].xpath("./*[1]").xpath("string(.)").get() if cadres else ""
        formats = set(re.findall(r"\bBO\s*(\d+)\b", header_text or "", re.I))
        if len(formats) != 1 or int(next(iter(formats))) < 1:
            raise ValueError(f"Match {match_id}: missing or ambiguous source series format")
        source_best_of = int(next(iter(formats)))

        async def process_game(game_id: str) -> dict:
            for attempt in range(1, GAME_FETCH_RETRIES + 1):
                try:
                    game_page_sel, pstats = await _gather_owned(
                        self.get_game_selector(game_id),
                        self.get_players_stats(game_id=game_id),
                    )
                    break
                except httpx.HTTPError:
                    if attempt == GAME_FETCH_RETRIES:
                        raise
                    await asyncio.sleep(attempt)
            identity = self._game_identity(game_page_sel, game_id)
            tstats = await self.get_team_stats(game_sel=game_page_sel, game_id=game_id)
            players = await self.get_players_in_game(game_sel=game_page_sel, game_id=game_id)
            for side in ("blue", "red"):
                for role, player in players[identity[f"{side}_id"]].items():
                    player["stats"] = pstats[side][role]
            return {
                "game_id": game_id,
                "match_id": str(match_id),
                "t1_id": identity["blue_id"],
                "t2_id": identity["red_id"],
                "t1_name": identity["blue_name"],
                "t2_name": identity["red_name"],
                "t1_win": identity["blue_win"],
                "t2_win": identity["red_win"],
                "draw": False,
                "t1_side": "blue",
                "t2_side": "red",
                "t1_players": players[identity["blue_id"]],
                "t2_players": players[identity["red_id"]],
                "t1_stats": tstats["blue"],
                "t2_stats": tstats["red"],
                "game_duration": tstats["gameDuration"],
                "source_game_ids": game_ids,
                "source_best_of": source_best_of,
                "source_identity": identity["raw"],
                "source_correction": identity["correction"],
            }

        return await _gather_owned(*(process_game(game_id) for game_id in game_ids))

    async def get_team_players(self, team_id: str) -> list[dict]:
        """Get the players in a team."""
        s = f"/teams/team-stats/{team_id}/split-ALL/tournament-ALL/"
        url = f"{GOLGG_URL}{s}"
        async with self.semaphore:
            response = await self.client.get(url)
        response.raise_for_status()

        html = response.content.decode("utf-8")

        sel = parsel.Selector(text=html)
        players_table = sel.css(".table_list")[-1]
        players = players_table.xpath("./tbody/tr")
        team_players = []
        for player in players:
            if len(player.css("td")) < 2:
                continue
            if len(team_players) >= 5:
                break
            player_tds = player.css("td")
            role = player_tds[0].css("::text").get().strip()
            player_link = player_tds[1].css("a")[0]
            href = player_link.css("::attr(href)").get()
            player_name = player_link.css("::text").get()
            player_id = re.search(r"player-stats/(\d+)/", href).group(1)
            team_players.append(
                {
                    "role": role,
                    "player_id": player_id,
                    "name": player_name,
                }
            )
        return team_players


async def main():
    game_id = "383"
    match_id = "2960"
    async with GolggScraper() as scrapper:
        games = await scrapper.get_games_in_match(match_id=match_id)
        with open("test_games.json", "w") as f:
            json.dump(games, f, indent=4)


if __name__ == "__main__":

    asyncio.run(main())
