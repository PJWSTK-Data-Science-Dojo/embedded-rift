import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import itertools
from datetime import date
import statistics

# Load your games list (assuming JSON format)
with open("games.json", "r") as f:
    games = json.load(f)

# Sort games by date (assuming date format "YYYY-MM-DD")
games.sort(key=lambda g: g["date"])

# Dictionary mapping player_id to a list of (date, rating)
player_rating_history = defaultdict(lambda: defaultdict(list))
# Dictionary for current ratings; starting rating set to 1500
current_ratings = defaultdict(lambda: 1500)

expected = lambda p1, p2: 1 / (1 + 10 ** ((p2 - p1) / 400))


# Instead of one current_ratings, keep two:
current_ratings_new = defaultdict(lambda: 1500)
current_ratings_old = defaultdict(lambda: 1500)

# Lists to collect pre‐match avg rating differences
diffs_new = []
diffs_old = []


def update_elo_new(team1_ids, team2_ids, t1_win, k=32):
    t1_avg = sum(current_ratings_new[p] for p in team1_ids) / len(team1_ids)
    t2_avg = sum(current_ratings_new[p] for p in team2_ids) / len(team2_ids)
    outcome = 1 if t1_win else 0
    for p in team1_ids:
        current_ratings_new[p] += k * (
            outcome - expected(current_ratings_new[p], t2_avg)
        )
    for p in team2_ids:
        current_ratings_new[p] += k * (
            (1 - outcome) - expected(current_ratings_new[p], t1_avg)
        )


def update_elo_old(team1_ids, team2_ids, t1_win, k=32):
    t1_avg = sum(current_ratings_old[p] for p in team1_ids) / len(team1_ids)
    t2_avg = sum(current_ratings_old[p] for p in team2_ids) / len(team2_ids)
    expected_t1 = 1 / (1 + 10 ** ((t2_avg - t1_avg) / 400))
    outcome = 1 if t1_win else 0
    for p in team1_ids:
        current_ratings_old[p] += k * (outcome - expected_t1)
    for p in team2_ids:
        current_ratings_old[p] += k * ((1 - outcome) - (1 - expected_t1))


# Process each game
for game in tqdm(games):
    # parse teams & result
    team1 = [game["t1_players"][r]["player_id"] for r in game["t1_players"]]
    team2 = [game["t2_players"][r]["player_id"] for r in game["t2_players"]]
    t1_win = game["t1_win"]

    # compute and record the pre‐match diff
    t1_new_avg = sum(current_ratings_new[p] for p in team1) / len(team1)
    t2_new_avg = sum(current_ratings_new[p] for p in team2) / len(team2)
    diffs_new.append(t1_new_avg - t2_new_avg)

    t1_old_avg = sum(current_ratings_old[p] for p in team1) / len(team1)
    t2_old_avg = sum(current_ratings_old[p] for p in team2) / len(team2)
    diffs_old.append(t1_old_avg - t2_old_avg)

    # update both systems
    update_elo_new(team1, team2, t1_win)
    update_elo_old(team1, team2, t1_win)
    game_date = pd.to_datetime(game["date"])
    for player_id in team1 + team2:
        player_rating_history[game_date.year][player_id].append(
            current_ratings_new[player_id]
        )


# Helper function to create a mapping from player_id to player_name
def get_player_names(games):
    mapping = {}
    for game in games:
        for role, player in game["t1_players"].items():
            mapping[player["player_id"]] = player["player_name"]
        for role, player in game["t2_players"].items():
            mapping[player["player_id"]] = player["player_name"]
    return mapping


player_names = get_player_names(games)


def get_top_10_each_year(player_rating_history):
    """
    Return a dictionary mapping each year to the top 10 players (player_id and rating)
    based on their final rating update within that year.
    """
    top_10_by_year_max = {}
    top_10_by_year_avg = {}
    all_years = player_rating_history.keys()
    min_year = min(all_years)
    max_year = max(all_years)

    for year in tqdm(range(min_year, max_year + 1)):
        players = player_rating_history[year]
        players_avg = {p: statistics.mean(ratings) for p, ratings in players.items()}
        players_max = {p: max(ratings) for p, ratings in players.items()}

        sorted_players_avg = sorted(
            players_avg.items(), key=lambda x: x[1], reverse=True
        )
        sorted_players_max = sorted(
            players_max.items(), key=lambda x: x[1], reverse=True
        )
        top_10_by_year_max[year] = sorted_players_max[:10]
        top_10_by_year_avg[year] = sorted_players_avg[:10]
    return top_10_by_year_max, top_10_by_year_avg


top_10_each_year_max, top_10_each_year_avg = get_top_10_each_year(player_rating_history)


for year, players in top_10_each_year_max.items():
    print(f"Year {year} Top 10 Players (Max Rating):")
    for player_id, rating in players:
        player_name = player_names.get(player_id, "Unknown")
        print(f"  Player ID: {player_id}, Name: {player_name} Rating: {rating:.2f}")


for year, players in top_10_each_year_avg.items():
    print(f"Year {year} Top 10 Players (Avg Rating):")
    for player_id, rating in players:
        player_name = player_names.get(player_id, "Unknown")
        print(f"  Player ID: {player_id}, Name: {player_name} Rating: {rating:.2f}")


# -----------------------------
# Plotting the Rating Trajectories
# -----------------------------

# Define your highlighted set (as strings of player IDs)
highlighted = {
    "1250",  # Showmaker
    # "1618",  # Czekolad
    "48",  # Faker
    "392",  # Peanut
    "1501",  # Inspierd
    "1075",  # TheShy
    "171",  # Uzi
    "470",  # Doinb
    # "5947", # Baus
    "392",
    "1629",  # Chovy
    "3247",  # Gumayusi
}

plt.figure(figsize=(12, 8))

# First, plot non-highlighted players in gray (with lower zorder)
for player_id, history in player_rating_history.items():
    if player_id in highlighted:
        continue  # Skip highlighted players for now
    df = pd.DataFrame(history, columns=["date", "rating"])
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    plt.plot(df["date"], df["rating"], color="gray", linewidth=1, alpha=0.3, zorder=1)


# Helper to check if a hex color is gray (R == G == B)
def is_gray(color):
    if color.startswith("#") and len(color) == 7:
        r, g, b = color[1:3], color[3:5], color[5:7]
        return r.lower() == g.lower() == b.lower()
    return False


# Get default color cycle and filter out gray colors
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
filtered_colors = [c for c in default_colors if not is_gray(c)]
color_cycle = itertools.cycle(filtered_colors)

# Plot highlighted players with distinct non-gray colors and a higher zorder.
for player_id in highlighted:
    if player_id not in player_rating_history:
        print(f"Player ID {player_id} not found in player_rating_history.")
        # Skip if player_id is not in player_rating_history
        continue
    history = player_rating_history[player_id]
    df = pd.DataFrame(history, columns=["date", "rating"])
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    color = next(color_cycle)
    print(f"Plotting {player_id} with color {color} and rating: {df["rating"]}")
    plt.plot(
        df["date"],
        df["rating"],
        label=player_names.get(player_id, player_id),
        linewidth=1,
        zorder=3,
        color=color,
    )

plt.title("Player Elo Rating Trajectories Over Time")
plt.xlabel("Date")
plt.ylabel("Elo Rating")
plt.legend()
plt.tight_layout()

diff_abs = [abs(diff) for diff in diffs_new]

abs_mean = statistics.mean(diff_abs)
abs_std = statistics.stdev(diff_abs)
print(f"Abs Mean: {abs_mean:.2f}, Std: {abs_std:.2f}")
new_mean = statistics.mean(diffs_new)
old_mean = statistics.mean(diffs_old)
new_std = statistics.stdev(diffs_new)
old_std = statistics.stdev(diffs_old)
print(f"New Elo Mean: {new_mean:.2f}, Std: {new_std:.2f}")
print(f"Old Elo Mean: {old_mean:.2f}, Std: {old_std:.2f}")

plt.figure(figsize=(10, 6))
plt.hist(
    diffs_new, bins=30, alpha=0.5, label="New‐Elo Δ", edgecolor="black", density=True
)
plt.hist(
    diffs_old, bins=30, alpha=0.5, label="Old‐Elo Δ", edgecolor="black", density=True
)
# plt.hist(diff_abs, bins=30, alpha=0.5, label="Abs Δ", edgecolor="black", density=True)
plt.title("Histogram of Team Avg‐Rating Differences\n(New vs. Old Elo Update)")
plt.xlabel("Team1 Avg Rating − Team2 Avg Rating")
plt.ylabel("Number of Games")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Compute Top 10 for Each Year
# -----------------------------
