from utils.scrapers import LoLScraper
import json
from dataclasses import dataclass, asdict
import tqdm

if __name__ == "__main__":
    # champion = "Dr. Mundo"
    # data = LoLScraper.get_champion_data(champion)
    champions = LoLScraper.get_all_champions()
    for champion in tqdm.tqdm(champions):
        data = LoLScraper.get_champion_data(champion)
        with open(f"data/champions/{champion}.json", "w") as f:
            json.dump(asdict(data), f, indent=4)
