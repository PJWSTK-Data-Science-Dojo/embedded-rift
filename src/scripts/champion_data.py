from utils.scrapers import LoLScraper
import json
from dataclasses import asdict
import tqdm

if __name__ == "__main__":
    champion = "Xin Zhao"
    # data = LoLScraper.get_champion_data(champion)
    scraper = LoLScraper()
    scraper.get_champion_data(champion)
    champions = scraper.get_all_champions()
    failed = []
    champions_data = {}
    for champion in tqdm.tqdm(champions):
        try:
            data = scraper.get_champion_data(champion)
            champions_data[champion] = asdict(data)

            with open(f"data/champions/{champion}.json", "w") as f:
                json.dump(asdict(data), f, indent=4)
        except Exception as e:
            print(f"Failed to fetch data for {champion}")
            print(e)
            failed.append(champion)

    with open("data/champions.json", "w") as f:
        json.dump(champions_data, f, indent=4)

    scraper.wiki.save_cache()
    with open("data/failed_champions.json", "w") as f:
        json.dump(failed, f, indent=4)
