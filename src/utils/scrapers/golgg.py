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


class GolggScraper:
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

    async def get_tournaments_in_season(self, season: int = 9) -> list[dict]:
        response = await self.client.post(
            GOLGG_TOURNAMENT_API,
            data={"season": f"S{season}"},
        )
        data = response.json()
        return data

    async def get_all_tournaments(self) -> list[dict]:
        result = []
        for season in range(2, 16):
            data = await self.get_tournaments_in_season(season)
            result.extend(data)
        return result

    async def get_matches_in_tournament(self, tournament_name: str) -> set[str]:
        """Process a single tournament using a dedicated page."""
        failed = []
        encoded_trname = quote(tournament_name)
        s = f"tournament-matchlist/{encoded_trname}/"

        url = f"{GOLGG_URL}/tournament/{s}"

        response = await self.client.get(url)
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
                score = row.css("td:nth-child(3)::text").get().strip()

                team_a_score, team_b_score = score.split("-")
                team_a_score = int(team_a_score.strip())
                team_b_score = int(team_b_score.strip())

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
                    "patch": patch,
                    "date": date,
                }
            except Exception as e:
                print("Error processing row:", e)
                failed.append(row.get())
                continue
            matches.append(data)
        with open("failed.html", "w") as f:
            f.write("\n".join(failed))
        return matches

    async def get_game_selector(self, game_id: str) -> parsel.Selector:
        s = f"/game/stats/{game_id}/page-game/"
        url = f"{GOLGG_URL}{s}"

        async with self.semaphore:
            response = await self.client.get(url)
        html = response.content.decode("utf-8")

        sel = parsel.Selector(text=html)
        return sel

    async def get_players_stats(self, game_id: str) -> dict[str, dict]:
        s = f"/game/stats/{game_id}/page-fullstats/"
        url = f"{GOLGG_URL}{s}"
        async with self.semaphore:
            response = await self.client.get(url)
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
            player_link = player_td.css("a")[1]
            href = player_link.css("::attr(href)").get()
            player_name = player_link.css("::text").get()
            player_id = re.search(r"player-stats/(\d+)/", href).group(1)
            player_data = {
                "player_id": player_id,
                "player_name": player_name,
            }
            role = INDEX_TO_ROLE.get(i, "UNKNOWN")
            team_1[role] = player_data

        team_2 = {}
        for i, player in enumerate(t2_players):
            player_td = player.css("td")[0]
            player_link = player_td.css("a")[1]
            href = player_link.css("::attr(href)").get()
            player_name = player_link.css("::text").get()

            player_id = re.search(r"player-stats/(\d+)/", href).group(1)
            player_data = {
                "player_id": player_id,
                "player_name": player_name,
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
        html = response.content.decode("utf-8")

        sel = parsel.Selector(text=html)
        navbar = sel.css("#gameMenuToggler")
        if not navbar:
            print("Couldn't find navbar in", url)
            return set()

        game_links = [
            el
            for el in navbar.css("li > a")
            if el.css("::text").get().strip().lower().startswith("game")
        ]
        match_table = sel.css(".col-cadre")[0]
        teams, *games_sel = match_table.xpath("./*")
        t1_link, t2_link = teams.css("a")
        t1_href = t1_link.css("::attr(href)").get()
        t2_href = t2_link.css("::attr(href)").get()
        pattern = r"teams/team-stats/(\d+)/"
        t1_id = re.search(pattern, t1_href).group(1)
        t2_id = re.search(pattern, t2_href).group(1)
        t1_name = t1_link.css("::text").get()
        t2_name = t2_link.css("::text").get()
        games = []
        for i, game_sel in enumerate(games_sel):
            team_1, _, team_2 = game_sel.xpath("./*")
            game_link = game_links[i]
            game_href = game_link.css("::attr(href)").get()
            pattern = r"game/stats/(\d+)/"
            game_id = re.search(pattern, game_href).group(1)
            t1_win = False
            if team_1.css(".text_victory"):
                t1_win = True

            game_sel = await self.get_game_selector(game_id)
            tstats = await self.get_team_stats(game_sel=game_sel)
            t1_side = "blue" if t1_id == tstats["blue_id"] else "red"
            t2_side = "red" if t1_side == "blue" else "blue"

            players = await self.get_players_in_game(game_sel=game_sel)
            pstats = await self.get_players_stats(game_id=game_id)
            for role, player in players[t1_id].items():
                player["stats"] = pstats[t1_side][role]

            for role, player in players[t2_id].items():
                player["stats"] = pstats[t2_side][role]

            games.append(
                {
                    "game_id": game_id,
                    "match_id": match_id,
                    "t1_id": t1_id,
                    "t2_id": t2_id,
                    "t1_name": t1_name,
                    "t2_name": t2_name,
                    "t1_win": t1_win,
                    "t2_win": not t1_win,
                    "t1_players": players[t1_id],
                    "t2_players": players[t2_id],
                    "t1_stats": tstats[t1_side],
                    "t2_stats": tstats[t2_side],
                    "game_duration": tstats["gameDuration"],
                }
            )

        return games


async def main():
    game_id = "383"
    match_id = "56264"
    async with GolggScraper() as scrapper:
        games = await scrapper.get_games_in_match(match_id=match_id)
        with open("test_games.json", "w") as f:
            json.dump(games, f, indent=4)


if __name__ == "__main__":

    asyncio.run(main())
