from playwright.sync_api import sync_playwright
import requests

GOLGG_URL = "https://gol.gg/tournament"
GOLGG_TURNAMENT_API = "https://gol.gg/tournament/ajax.trlist.php"


class GolggScraper:
    def __init__(self, tournament_id):
        self.tournament_id = tournament_id
        self.url = f"{GOLGG_URL}/{tournament_id}"

    def __enter__(self):
        pass

    def get_teams(self):
        pass

    def get_matches(self):
        pass

    def get_players(self):
        pass

    def get_stats(self):
        pass


if __name__ == "__main__":
    tournaments_count = 0
    matches_count = 0
    for season in range(9, 16):
        body = {
            "season": f"S{season}",
        }

        response = requests.post(GOLGG_TURNAMENT_API, data=body)
        data = response.json()
        tournaments_count += len(data)
        for tournament in data:
            nbgames = tournament["nbgames"]
            matches_count += int(nbgames)

    print(f"Total tournaments: {tournaments_count}")
    print(f"Total matches: {matches_count}")
