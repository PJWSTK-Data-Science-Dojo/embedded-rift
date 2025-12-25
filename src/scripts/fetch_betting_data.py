#!/usr/bin/env python3
"""
Fetch League of Legends betting data from STS.pl.

This script fetches live LoL betting matches from STS using HTML scraping
and saves the results to a JSON file.

Usage:
    python src/scripts/fetch_betting_data.py
"""

import json
import sys
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.scrapers import LoLScraper


def serialize_datetime(obj):
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def serialize_market(market):
    """Convert BettingMarket to dict for JSON serialization."""
    return {
        "market_type": market.market_type,
        "market_type_id": market.market_type_id,
        "description": market.description,
        "odds": [
            {
                "outcome": odds.outcome,
                "odds": odds.odds,
                "is_active": odds.is_active,
            }
            for odds in market.odds_list
        ],
    }


def main():
    """Fetch and display betting data."""
    print("\n" + "=" * 70)
    print("STS.pl League of Legends Betting Data - Scraper")
    print("=" * 70 + "\n")

    # Initialize scraper
    print("Initializing LoLScraper...")
    try:
        scraper = LoLScraper()
        print("✓ LoLScraper initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize LoLScraper: {e}\n")
        return 1

    # Fetch matches
    print("Fetching League of Legends matches from STS.pl...")
    try:
        matches = scraper.get_betting_matches()
        print(f"✓ Fetched {len(matches)} matches\n")
    except Exception as e:
        print(f"✗ Failed to fetch matches: {e}\n")
        return 1

    # Display match information
    if matches:
        print("=" * 70)
        print("Matches Found")
        print("=" * 70 + "\n")

        for i, match in enumerate(matches, 1):
            print(f"{i}. {match.team1} vs {match.team2}")
            print(f"   Time: {match.start_time}")
            print(f"   Tournament: {match.tournament}\n")

            # Show odds
            if match.markets:
                for market_type, market in match.markets.items():
                    if market.odds_list:
                        print(f"   {market.description}:")
                        for odds in market.odds_list:
                            print(f"     • {odds.outcome}: {odds.odds}")
                        print()

        # Save to file
        print("=" * 70)
        print("Saving Data")
        print("=" * 70 + "\n")

        data_dir = Path(__file__).parent.parent.parent / "data"
        data_dir.mkdir(exist_ok=True)

        try:
            # Convert matches to serializable format
            matches_data = []
            for match in matches:
                match_dict = {
                    "match_id": match.match_id,
                    "team1": match.team1,
                    "team2": match.team2,
                    "tournament": match.tournament,
                    "start_time": match.start_time.isoformat(),
                    "is_live": match.is_live,
                    "match_url": match.match_url,
                    "markets": {
                        name: serialize_market(market)
                        for name, market in match.markets.items()
                    },
                }
                matches_data.append(match_dict)

            # Save to JSON
            output_file = data_dir / "sts_lol_betting_matches.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(matches_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Saved {len(matches)} matches to: {output_file}\n")

        except Exception as e:
            print(f"✗ Failed to save data: {e}\n")
            return 1

        return 0

    else:
        print("⚠ No matches found\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
