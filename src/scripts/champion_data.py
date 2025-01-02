from utils.scrapers import LoLScraper
import json
from dataclasses import dataclass, asdict
import tqdm

if __name__ == "__main__":
    # champion = "Dr. Mundo"
    # data = LoLScraper.get_champion_data(champion)
    scraper = LoLScraper()
    champions = scraper.get_all_champions()
    failed = []
    for champion in tqdm.tqdm(champions):
        try:
            data = scraper.get_champion_data(champion)
            with open(f"data/champions/{champion}.json", "w") as f:
                json.dump(asdict(data), f, indent=4)

        except Exception as e:
            print(f"Failed to fetch data for {champion}")
            failed.append(champion)

    scraper.wiki.save_cache()
    with open("data/failed_champions.json", "w") as f:
        json.dump(failed, f, indent=4)
