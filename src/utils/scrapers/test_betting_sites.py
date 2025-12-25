#!/usr/bin/env python3
"""
Unified test script for all League of Legends betting scrapers.
Tests Superbet, STS, and Betclic scrapers to ensure consistent data format.
Saves results to data/all_bettings_sites.json.
"""

import sys
import os
import json
from datetime import datetime
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.scrapers import LoLScraper


def format_match(match):
    """Format a match for display."""
    return f"{match.team1:<20} vs {match.team2:<20} @ {match.start_time.strftime('%H:%M')} ({match.tournament:<12})"


def validate_match(match, scraper_name):
    """Validate that a match has all required fields."""
    errors = []

    # Check required fields
    if not match.match_id:
        errors.append("match_id is empty")
    if not match.team1:
        errors.append("team1 is empty")
    if not match.team2:
        errors.append("team2 is empty")
    if not match.tournament:
        errors.append("tournament is empty")
    if not match.start_time:
        errors.append("start_time is empty")
    if not match.markets:
        errors.append("markets is empty")

    # Check odds
    if match.markets:
        for market_name, market in match.markets.items():
            if not market.odds_list:
                errors.append(f"Market '{market_name}' has no odds")
            else:
                for odd in market.odds_list:
                    if odd.odds <= 0:
                        errors.append(f"Invalid odds value: {odd.odds}")
                    if not odd.outcome:
                        errors.append(f"Odds missing outcome")

    return errors


def test_scraper(scraper, scraper_name, method_name):
    """Test a single scraper."""
    print(f"\n{'=' * 100}")
    print(f"Testing {scraper_name.upper()}")
    print(f"{'=' * 100}")

    try:
        print(f"Fetching matches from {scraper_name} (this may take 10-20 seconds)...")
        method = getattr(scraper, method_name)
        matches = method()

        print(f"✓ Retrieved {len(matches)} matches from {scraper_name}")

        if not matches:
            print(f"⚠ WARNING: No matches returned from {scraper_name}")
            return {"name": scraper_name, "count": 0, "matches": []}

        # Display sample matches
        print(f"\n{scraper_name.upper()} - Sample Matches (first 5):")
        print(f"{'-' * 100}")
        print(f"{'Team 1':<20} vs {'Team 2':<20} Time    Tournament")
        print(f"{'-' * 100}")

        sample_matches = matches[:5]
        for match in sample_matches:
            print(format_match(match))

        # Validate matches
        print(f"\n{scraper_name.upper()} - Data Validation (first 5 matches):")
        print(f"{'-' * 100}")

        all_valid = True
        for i, match in enumerate(sample_matches, 1):
            errors = validate_match(match, scraper_name)
            if errors:
                print(f"✗ Match {i} ({match.team1} vs {match.team2}): {', '.join(errors)}")
                all_valid = False
            else:
                print(f"✓ Match {i} ({match.team1} vs {match.team2}): Valid")

        # Show odds
        print(f"\n{scraper_name.upper()} - Odds Format Check:")
        print(f"{'-' * 100}")
        for i, match in enumerate(sample_matches[:2], 1):
            print(f"\nMatch {i}: {match.team1} vs {match.team2}")
            for market_name, market in match.markets.items():
                print(f"  Market: {market_name}")
                for odd in market.odds_list:
                    print(f"    {odd.outcome}: {odd.odds} (type: {type(odd.odds).__name__})")

        return {
            "name": scraper_name,
            "count": len(matches),
            "matches": matches,
            "valid": all_valid,
        }

    except Exception as e:
        print(f"✗ Error testing {scraper_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"name": scraper_name, "count": 0, "matches": [], "error": str(e)}


def compare_scrapers(results):
    """Compare results from all scrapers."""
    print(f"\n\n{'=' * 100}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 100}")

    print(f"\n{'Scraper':<15} {'Matches':<12} {'Status':<12}")
    print(f"{'-' * 100}")

    total_matches = 0
    for result in results:
        status = "✓ Valid" if result.get("valid") else "✗ Invalid"
        if "error" in result:
            status = f"✗ Error: {result['error'][:40]}"
        print(f"{result['name']:<15} {result['count']:<12} {status:<12}")
        total_matches += result["count"]

    print(f"\nTotal matches across all scrapers: {total_matches}")

    # Compare data consistency
    if len(results) > 1 and all("matches" in r and r["matches"] for r in results):
        print(f"\n{'-' * 100}")
        print("Data Format Consistency Check:")
        print(f"{'-' * 100}")

        first_result = results[0]
        first_match = first_result["matches"][0]

        # Check all results have same field types
        consistent = True
        for result in results[1:]:
            if not result.get("matches"):
                continue

            match = result["matches"][0]

            # Compare field types
            if (
                type(first_match.match_id) != type(match.match_id)
                or type(first_match.start_time) != type(match.start_time)
                or type(first_match.markets) != type(match.markets)
            ):
                print(
                    f"✗ {result['name']}: Field type mismatch with {first_result['name']}"
                )
                consistent = False

        if consistent:
            print("✓ All scrapers use consistent data types")
            print(f"  - match_id: {type(first_match.match_id).__name__}")
            print(f"  - team1: {type(first_match.team1).__name__}")
            print(f"  - team2: {type(first_match.team2).__name__}")
            print(f"  - tournament: {type(first_match.tournament).__name__}")
            print(f"  - start_time: {type(first_match.start_time).__name__}")
            print(f"  - markets: {type(first_match.markets).__name__}")
            print(f"  - odds values: {type(list(first_match.markets.values())[0].odds_list[0].odds).__name__}")


def save_results_to_json(results, filename="data/all_bettings_sites.json"):
    """Save scraper results to a JSON file."""
    print(f"\nSaving results to {filename}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Process results for JSON serialization
    json_data = []
    for result in results:
        scraper_data = {
            "scraper": result["name"],
            "matches": []
        }
        
        for match in result["matches"]:
            # Convert dataclass to dict
            match_dict = asdict(match)
            # Handle datetime serialization
            if isinstance(match_dict["start_time"], datetime):
                match_dict["start_time"] = match_dict["start_time"].isoformat()
            
            scraper_data["matches"].append(match_dict)
        
        json_data.append(scraper_data)
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved successfully to {filename}")
    except Exception as e:
        print(f"✗ Failed to save results to JSON: {e}")


def main():
    """Run all tests."""
    print("=" * 100)
    print("UNIFIED LEAGUE OF LEGENDS BETTING SCRAPERS TEST")
    print("=" * 100)
    print("\nTesting all three betting scrapers:")
    print("  1. Superbet (API-based)")
    print("  2. STS (HTML parsing with Playwright)")
    print("  3. Betclic (HTML parsing with Camoufox)")

    # Initialize scraper
    print("\nInitializing LoLScraper...")
    try:
        scraper = LoLScraper()
        print("✓ LoLScraper initialized")
    except Exception as e:
        print(f"✗ Failed to initialize LoLScraper: {e}")
        return False

    # Test each scraper
    results = []

    # Test Superbet (fastest, should be first)
    superbet_result = test_scraper(scraper, "superbet", "get_betting_matches_superbet")
    results.append(superbet_result)

    # Test STS
    sts_result = test_scraper(scraper, "sts", "get_betting_matches")
    results.append(sts_result)

    # Test Betclic
    betclic_result = test_scraper(scraper, "betclic", "get_betting_matches_betclic")
    results.append(betclic_result)

    # Compare all results
    compare_scrapers(results)
    
    # Save results to JSON
    save_results_to_json(results)

    # Final verdict
    print(f"\n{'=' * 100}")
    print("FINAL VERDICT")
    print(f"{'=' * 100}")

    all_working = all(r["count"] > 0 for r in results)
    all_valid = all(r.get("valid", True) for r in results)

    if all_working:
        print("✓ ALL SCRAPERS WORKING!")
        if all_valid:
            print("✓ All data validated successfully")
        else:
            print("⚠ Some validation errors detected")
    else:
        print("✗ Some scrapers not working")
        return False

    print(f"\n{'-' * 100}")
    print("Summary:")
    print(f"  ✓ Superbet: {superbet_result['count']} matches")
    print(f"  ✓ STS: {sts_result['count']} matches")
    print(f"  ✓ Betclic: {betclic_result['count']} matches")
    print(f"  ✓ Total: {sum(r['count'] for r in results)} matches across all scrapers")
    print(f"{'=' * 100}\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
