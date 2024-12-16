from playwright.sync_api import sync_playwright
import requests
import json

GOLGG_URL = "https://gol.gg/tournament"
GOLGG_TURNAMENT_API = "https://gol.gg/tournament/ajax.trlist.php"


class GolggScraper:
    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch()
        self.page = self.browser.new_page()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.browser.close()
        self.playwright.stop()

    def get_tournaments(self) -> list[dict]:
        tournaments = []
        for season in range(9, 16):
            body = {
                "season": f"S{season}",
            }
            response = requests.post(GOLGG_TURNAMENT_API, data=body)
            data = response.json()
            clean_data = [{**entry, "nbgames": int(entry["nbgames"])} for entry in data]
            tournaments.extend(clean_data)

        return tournaments

    def get_tournaments_in_season(self, season: int = 9) -> list[dict]:
        body = {
            "season": f"S{season}",
        }
        response = requests.post(GOLGG_TURNAMENT_API, data=body)
        data = response.json()

        clean_data = [{**entry, "nbgames": int(entry["nbgames"])} for entry in data]

        return clean_data

    def get_teams(self):
        pass

    def get_matches(self):
        pass

    def get_players(self):
        pass

    def get_stats(self):
        pass


if __name__ == "__main__":
    with GolggScraper() as scraper:
        tournaments = scraper.get_tournaments()
        with open("tournaments.json", "w") as f:
            json.dump(tournaments, f, indent=4)

    print(f"Total tournaments: {len(tournaments)}")
