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


class GolggScraper:
    def __init__(self, max_pages: int = 20):
        self.semaphore = asyncio.Semaphore(max_pages)
        self.client: httpx.AsyncClient | None = None

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

        response = await self.client.get(url)
        response.raise_for_status()
        html = response.content.decode("utf-8")

        sel = parsel.Selector(text=html)
        tables = sel.css(".table_list")
        matches_table = next(
            (table for table in tables if "data-sort" in table.attrib), None
        )

        if not matches_table:
            print("Couldn't find games table in", url)
            return []

        # Extract links and match IDs
        rows = matches_table.css("tbody tr")
        matches = []

        for row in rows:
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
                score = row.css("td:nth-child(3)::text").get()
                if not score or "-" not in score:
                    continue
                score = score.strip()

                score_left, score_right = score.split("-")
                score_left = int(score_left.strip())
                score_right = int(score_right.strip())

                # GOL.gg renders the score as winner-loser, not always as
                # team_a-team_b. Normalize it back to the order from
                # "team_a vs team_b" so downstream code can trust t1/t2_score.
                if won == team_a:
                    team_a_score = score_left
                    team_b_score = score_right
                elif won == team_b:
                    team_a_score = score_right
                    team_b_score = score_left
                else:
                    team_a_score = score_left
                    team_b_score = score_right

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
        rows = table.xpath("./tr")
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

    async def get_players_in_game(self, game_sel: parsel.Selector) -> dict[str, dict]:
        """Get the players in a game."""

        team_table = game_sel.css(".col-cadre")[0]
        team_row = team_table.xpath("./*")[1]
        team_1_block, team_2_block = team_row.xpath("./*")
        team_1_info = team_1_block.xpath("./*")[0]
        team_2_info = team_2_block.xpath("./*")[0]
        team_1_link = team_1_info.css("a::attr(href)").get()
        team_2_link = team_2_info.css("a::attr(href)").get()
        team_1_id = re.search(r"teams/team-stats/(\d+)/", team_1_link).group(1)
        team_2_id = re.search(r"teams/team-stats/(\d+)/", team_2_link).group(1)
        t1_players_table, t2_players_table = game_sel.css(".playersInfosLine")
        t1_players = t1_players_table.xpath("./tr")
        t2_players = t2_players_table.xpath("./tr")
        team_1 = {}
        for i, player in enumerate(t1_players):
            player_td = player.css("td")[0]
            champion_data = extract_champion_from_player_row(player)
            player_link = player_td.css("a")[1]
            href = player_link.css("::attr(href)").get()
            player_name = player_link.css("::text").get()
            player_id = re.search(r"player-stats/(\d+)/", href).group(1)
            player_data = {
                "player_id": player_id,
                "player_name": player_name,
                **champion_data,
            }
            role = INDEX_TO_ROLE.get(i, "UNKNOWN")
            team_1[role] = player_data

        team_2 = {}
        for i, player in enumerate(t2_players):
            player_td = player.css("td")[0]
            champion_data = extract_champion_from_player_row(player)
            player_link = player_td.css("a")[1]
            href = player_link.css("::attr(href)").get()
            player_name = player_link.css("::text").get()

            player_id = re.search(r"player-stats/(\d+)/", href).group(1)
            player_data = {
                "player_id": player_id,
                "player_name": player_name,
                **champion_data,
            }
            role = INDEX_TO_ROLE.get(i, "UNKNOWN")
            team_2[role] = player_data

        return {
            team_1_id: team_1,
            team_2_id: team_2,
        }

    async def get_team_stats(self, game_sel: parsel.Selector) -> dict[str, dict]:
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
        bteam_id = bteam.css("a::attr(href)").get().strip()
        rteam_id = rteam.css("a::attr(href)").get().strip()
        bteam_id = re.search(r"teams/team-stats/(\d+)/", bteam_id).group(1)
        rteam_id = re.search(r"teams/team-stats/(\d+)/", rteam_id).group(1)
        result["blue_id"] = bteam_id
        result["red_id"] = rteam_id

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

    async def get_games_in_match(self, match_id):
        """Get the games ids in a match."""
        s = f"/game/stats/{match_id}/page-summary/"
        url = f"{GOLGG_URL}{s}"
        async with self.semaphore:
            response = await self.client.get(url)
        response.raise_for_status()
        html = response.content.decode("utf-8")

        sel = parsel.Selector(text=html)
        navbar = sel.css("#gameMenuToggler")
        if not navbar:
            print("Couldn't find navbar in", url)
            return []

        game_links = [
            el
            for el in navbar.css("li > a")
            if el.css("::text").get().strip().lower().startswith("game")
        ]
        match_tables = sel.css(".col-cadre")
        if not match_tables:
            print("Couldn't find match table in", url)
            return []
        match_table = match_tables[0]
        teams, *games_sel = match_table.xpath("./*")
        t1_link, t2_link = teams.css("a")
        t1_href = t1_link.css("::attr(href)").get()
        t2_href = t2_link.css("::attr(href)").get()
        pattern = r"teams/team-stats/(\d+)/"
        t1_id = re.search(pattern, t1_href).group(1)
        t2_id = re.search(pattern, t2_href).group(1)
        t1_name = t1_link.css("::text").get()
        t2_name = t2_link.css("::text").get()
        async def process_game(i: int, game_sel: parsel.Selector) -> dict | None:
            for attempt in range(1, GAME_FETCH_RETRIES + 1):
                try:
                    return await process_game_attempt(i, game_sel)
                except Exception as e:
                    print(
                        f"Error processing game {i + 1} in match {match_id} "
                        f"attempt {attempt}/{GAME_FETCH_RETRIES}: {e}"
                    )
                    if attempt < GAME_FETCH_RETRIES:
                        await asyncio.sleep(attempt)
            return None

        async def process_game_attempt(
            i: int, game_sel: parsel.Selector
        ) -> dict | None:
            if i >= len(game_links):
                print(f"Missing game link {i + 1} for match {match_id}")
                return None

            team_1, _, team_2 = game_sel.xpath("./*")
            game_link = game_links[i]
            game_href = game_link.css("::attr(href)").get()
            pattern = r"game/stats/(\d+)/"
            game_match = re.search(pattern, game_href or "")
            if not game_match:
                print("Couldn't extract game id from", game_href)
                return None
            game_id = game_match.group(1)
            t1_win = bool(team_1.css(".text_victory"))

            game_page_sel, pstats = await asyncio.gather(
                self.get_game_selector(game_id),
                self.get_players_stats(game_id=game_id),
            )
            tstats = await self.get_team_stats(game_sel=game_page_sel)
            t1_side = "blue" if t1_id == tstats["blue_id"] else "red"
            t2_side = "red" if t1_side == "blue" else "blue"

            players = await self.get_players_in_game(game_sel=game_page_sel)
            for role, player in players[t1_id].items():
                player["stats"] = pstats[t1_side][role]

            for role, player in players[t2_id].items():
                player["stats"] = pstats[t2_side][role]

            return {
                "game_id": game_id,
                "match_id": match_id,
                "t1_id": t1_id,
                "t2_id": t2_id,
                "t1_name": t1_name,
                "t2_name": t2_name,
                "t1_win": t1_win,
                "t2_win": not t1_win,
                "draw": False,
                "t1_side": t1_side,
                "t2_side": t2_side,
                "t1_players": players[t1_id],
                "t2_players": players[t2_id],
                "t1_stats": tstats[t1_side],
                "t2_stats": tstats[t2_side],
                "game_duration": tstats["gameDuration"],
            }

        game_results = await asyncio.gather(
            *(process_game(i, game_sel) for i, game_sel in enumerate(games_sel))
        )
        games = [game for game in game_results if game]

        return games

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
