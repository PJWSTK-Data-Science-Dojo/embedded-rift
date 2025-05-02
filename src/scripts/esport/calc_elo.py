import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from trueskill import Rating, rate
from openskill.models import PlackettLuce
import matplotlib.pyplot as plt
import openskill
from scipy.stats import norm
import math
import csv
import trueskill
from utils.rankings._elo import expected_win_elo, update_elo, update_team_elo
from utils.rankings._os import update_openskill
from utils.rankings._ts import team_rating, update_trueskill, expected_trueskill_win
import statistics as stat

model = PlackettLuce()
elo_ratings = defaultdict(lambda: 1500)
trueskill_ratings = defaultdict(lambda: Rating())
openskill_ratings = defaultdict(lambda: model.rating())
THRESHOLD = 0.5
trueskill.DRAW_PROBABILITY = 0.0


def plot_roc_auc(y_true, elo_probs, ts_probs, os_probs):
    """
    Plots ROC curves for three models and prints their AUCs.

    y_true:    list of 0/1 true outcomes (1 = team1 won)
    elo_probs: list of predicted P(team1 wins) from ELO
    ts_probs:  list of predicted P(team1 wins) from TrueSkill
    os_probs:  list of predicted P(team1 wins) from OpenSkill
    """
    plt.figure(figsize=(8, 6))
    for probs, name in (
        (elo_probs, "ELO"),
        (ts_probs, "TrueSkill"),
        (os_probs, "OpenSkill"),
    ):
        fpr, tpr, _ = roc_curve(y_true, probs)
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {model_auc:.3f})")

    # random‐guess line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")

    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_final_ratings(
    ratings_dict, title, name_map: dict[str, str], top_n=10, getter=lambda r: r
):
    ratings = [(p, getter(r)) for p, r in ratings_dict.items()]
    ratings.sort(key=lambda x: x[1], reverse=True)
    top_players = ratings[:top_n]

    names = [name_map[p] for p, _ in top_players]
    values = [v for _, v in top_players]

    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], values[::-1])
    plt.xlabel("Rating")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"viz/{title}.png")


def get_id_to_name_mapping(games):
    mapping = {}
    for game in games:
        for role, player in game["t1_players"].items():
            mapping[player["player_id"]] = player["player_name"]
        for role, player in game["t2_players"].items():
            mapping[player["player_id"]] = player["player_name"]
    return mapping


def get_name_to_id_mapping(games):
    mapping = {}
    for game in games:
        for role, player in game["t1_players"].items():
            mapping[player["player_name"]] = player["player_id"]
        for role, player in game["t2_players"].items():
            mapping[player["player_name"]] = player["player_id"]
    return mapping


def save_ratings(name_map, player_game_count):
    rating_path = Path("ratings.csv")
    data = []
    for p in elo_ratings:
        player_data = {
            "player_id": p,
            "player_name": name_map[p],
            "games_played": player_game_count[p],
            "elo_rating": elo_ratings[p],
            "trueskill_mu": trueskill_ratings[p].mu,
            "trueskill_sigma": trueskill_ratings[p].sigma,
            "openskill_mu": openskill_ratings[p].mu,
            "openskill_sigma": openskill_ratings[p].sigma,
        }

        data.append(player_data)

    with open(rating_path, "w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter=";")
        csv_writer.writeheader()
        csv_writer.writerows(data)


def count_player_games(games):
    """
    Count the number of games each player participated in.
    Returns a defaultdict mapping player_id -> number of games.
    """
    game_counter = defaultdict(int)
    for game in tqdm(games):
        # Count games for team1 players
        for role, player in game["t1_players"].items():
            game_counter[player["player_id"]] += 1
        # Count games for team2 players
        for role, player in game["t2_players"].items():
            game_counter[player["player_id"]] += 1
    return game_counter


def plot_rating_diff_histogram(win_diffs, title, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(win_diffs, bins=100, alpha=0.6)
    plt.title(title)
    plt.xlabel("Rating Difference (T1 - T2)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"viz/{filename}.png")


def plot_percentage_accuracy(
    perc_count: defaultdict, perc_sum: defaultdict, num_of_games: int, name: str = "elo"
):
    """
    Plots a histogram of predicted confidence percentages and the accuracy for each percentage bucket.

    Parameters:
    - ts_perc_count: dict mapping predicted percentage (float) to count of predictions
    - ts_perc_sum: dict mapping predicted percentage (float) to count of correct predictions
    - num_of_games: int total number of predictions
    """
    # Sort percentage buckets
    percents = sorted(perc_count.keys())
    counts = [perc_count[p] / num_of_games for p in percents]
    accuracies = [perc_sum.get(p, 0) / perc_count[p] for p in percents]

    # Histogram of counts
    plt.figure()
    plt.bar(percents, counts)
    plt.xlabel("Predicted Percentage")
    plt.ylabel("% of Population")
    plt.title(f"Histogram of Predicted Confidence Percentages ({name})")

    # Plot of accuracy per percentage
    plt.figure()
    plt.plot(percents, accuracies, marker="o")
    plt.xlabel("Predicted Percentage")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy per Predicted Percentage {name}")
    plt.ylim(0.45, 1)
    plt.xlim(50, 100)

    # 5%-interval ticks
    # plt.xticks(np.arange(50, 101, 5))
    plt.yticks(np.arange(0.5, 1.01, 0.05))

    # grid on major ticks
    plt.grid(which="major", linestyle="--", linewidth=0.5)

    # Print overall accuracy
    total_correct = sum(perc_sum.values())
    overall_accuracy = total_correct / num_of_games if num_of_games > 0 else 0
    print(f"Overall accuracy: {overall_accuracy:.2%} ({total_correct}/{num_of_games})")


elo_diffs = []
elo_pred = 0
count_elo = 0
elo_perc_count = defaultdict(int)
elo_perc_sum = defaultdict(int)


def elo_rating(t1, t2, t1_win):
    global elo_pred, count_elo
    r1, r2 = [elo_ratings[p] for p in t1], [elo_ratings[p] for p in t2]
    t1_avg = sum(r1) / len(r1)
    t2_avg = sum(r2) / len(r2)
    t1_win_pred = expected_win_elo(t1_avg, t2_avg)
    if t1_win_pred > THRESHOLD:
        elo_pred += 1 if t1_win else 0
        count_elo += 1
    elif t1_win_pred < (1 - THRESHOLD):
        elo_pred += 1 if not t1_win else 0
        count_elo += 1

    elo_diff = t1_avg - t2_avg
    elo_diffs.append(elo_diff)

    t1_up, t2_up = update_team_elo(r1, r2, t1_win)
    for player_id, new_ranking in zip(t1 + t2, t1_up + t2_up):
        elo_ratings[player_id] = new_ranking

    wp = t1_win_pred
    tw = t1_win
    if t1_win_pred < 0.5:
        wp = 1 - t1_win_pred
        tw = not t1_win
    wp1 = round(wp * 100)

    elo_perc_count[wp1] += 1
    elo_perc_sum[wp1] += 1 if tw else 0

    return t1_win_pred


ts_perc_count = defaultdict(int)
ts_perc_sum = defaultdict(int)
ts_diffs = []
ts_pred = 0
count_ts = 0


def ts_rating(t1, t2, t1_win):
    global ts_pred, count_ts
    r1, r2 = [trueskill_ratings[p] for p in t1], [trueskill_ratings[p] for p in t2]
    t1_win_pred = expected_trueskill_win(r1, r2)

    if t1_win_pred > THRESHOLD:
        ts_pred += 1 if t1_win else 0
        count_ts += 1
    elif t1_win_pred < (1 - THRESHOLD):
        ts_pred += 1 if not t1_win else 0
        count_ts += 1

    ts_mu1, ts_var1 = team_rating(r1)
    ts_mu2, ts_var2 = team_rating(r2)
    ts_denom = math.sqrt(len(r1) * trueskill.BETA**2 + ts_var1 + ts_var2)
    ts_diff = (ts_mu1 - ts_mu2) / ts_denom

    ts_diffs.append(ts_diff)

    t1_up, t2_up = update_trueskill(r1, r2, t1_win)
    for player_id, new_ranking in zip(t1 + t2, t1_up + t2_up):
        trueskill_ratings[player_id] = new_ranking

    wp = t1_win_pred
    tw = t1_win
    if t1_win_pred < 0.5:
        wp = 1 - t1_win_pred
        tw = not t1_win
    wp1 = round(t1_win_pred * 100)

    ts_perc_count[wp1] += 1
    ts_perc_sum[wp1] += 1 if tw else 0

    return t1_win_pred


os_diffs = []
os_pred = 0
os_perc_count = defaultdict(int)
os_perc_sum = defaultdict(int)
count_os = 0


def os_rating(t1, t2, t1_win):
    global os_pred, count_os
    r1, r2 = [openskill_ratings[p] for p in t1], [openskill_ratings[p] for p in t2]
    wp1, wp2 = model.predict_win([r1, r2])

    if wp1 > THRESHOLD:
        os_pred += 1 if t1_win else 0
        count_os += 1
    elif wp2 > THRESHOLD:
        os_pred += 1 if not t1_win else 0
        count_os += 1

    os_mu1, os_var1 = team_rating(r1)
    os_mu2, os_var2 = team_rating(r2)
    os_denom = math.sqrt(len(r1) * model.beta**2 + os_var1 + os_var2)
    os_diff = (os_mu1 - os_mu2) / os_denom
    os_diffs.append(os_diff)
    t1_up, t2_up = update_openskill(r1, r2, t1_win, model=model)
    for player_id, new_ranking in zip(t1 + t2, t1_up + t2_up):
        openskill_ratings[player_id] = new_ranking

    wp = wp1
    tw = t1_win
    if wp1 < wp2:
        wp = wp2
        tw = not t1_win

    max_wp = round(wp * 100)
    os_perc_count[max_wp] += 1
    os_perc_sum[max_wp] += 1 if tw else 0

    return wp1


def main():
    global elo_ratings, trueskill_ratings, openskill_ratings
    global elo_diffs, ts_diffs, os_diffs
    global elo_pred, ts_pred, os_pred
    games_path = Path("games.json")
    if not games_path.exists():
        print("Games file not found, please download it first.")
        return

    with open(games_path, "r") as f:
        games = json.load(f)

    games = sorted(games, key=lambda g: g["date"])
    name_map = get_id_to_name_mapping(games)
    player_game_count = count_player_games(games)
    y_true = []
    elo_probs = []
    ts_probs = []
    os_probs = []
    for game in tqdm(games, desc="Processing games"):
        t1 = [game["t1_players"][role]["player_id"] for role in game["t1_players"]]
        t2 = [game["t2_players"][role]["player_id"] for role in game["t2_players"]]
        t1_win = game["t1_win"]
        t1_name = game["t1_name"]
        t2_name = game["t2_name"]

        p_elo = elo_rating(t1, t2, t1_win)
        p_ts = ts_rating(t1, t2, t1_win)
        p_os = os_rating(t1, t2, t1_win)
        y_true.append(1 if t1_win else 0)

        elo_probs.append(p_elo)
        ts_probs.append(p_ts)
        os_probs.append(p_os)
        # if abs(elo_diff) > 300 or abs(ts_diff) > 3 or abs(os_diff) > 3:
        #     print(
        #         f"Game: {t1_name} vs {t2_name} (Won: {t1_name if t1_win else t2_name}) | Elo diff: {elo_diff:.2f} | TrueSkill diff: {ts_diff:.2f} | OpenSkill diff: {os_diff:.2f}"
        #     )
    print("len(y_true)   =", len(y_true))
    print("len(elo_probs)=", len(elo_probs))
    print("len(ts_probs) =", len(ts_probs))
    print("len(os_probs) =", len(os_probs))
    print(f"Elo prediction accuracy: {elo_pred / count_elo:.2%}")
    print(f"TrueSkill prediction accuracy: {ts_pred / count_ts:.2%}")
    print(f"OpenSkill prediction accuracy: {os_pred / count_os:.2%}")
    print(f"Total Elo games: {count_elo} ({count_elo / len(games) * 100:.2f}%)")
    print(f"Total TrueSkill games: {count_ts} ({count_ts / len(games) * 100:.2f}%)")
    print(f"Total OpenSkill games: {count_os} ({count_os / len(games) * 100:.2f}%)")

    name_to_id = get_name_to_id_mapping(games)
    team_1 = "T1 Academy"
    team_2 = "Nongshim Esports Academy"
    t1 = ["Haetae", "Vincenzo", "Poby", "Cypher", "Cloud"]
    t2 = ["Kangin", "Sylvie", "Calix", "Vital", "Crack"]
    t1 = [name_to_id[name] for name in t1]
    t2 = [name_to_id[name] for name in t2]
    r1 = [elo_ratings[p] for p in t1]
    r2 = [elo_ratings[p] for p in t2]
    avg_elo1 = sum(r1) / len(r1)
    avg_elo2 = sum(r2) / len(r2)
    ep = expected_win_elo(avg_elo1, avg_elo2)
    print(f"Elo prediction for {team_1} vs {team_2}: {ep:.2f}")
    r1, r2 = [trueskill_ratings[p] for p in t1], [trueskill_ratings[p] for p in t2]
    tp = expected_trueskill_win(r1, r2)
    print(f"TrueSkill prediction for {team_1} vs {team_2}: {tp:.2f}")
    r1, r2 = [openskill_ratings[p] for p in t1], [openskill_ratings[p] for p in t2]
    op, _ = model.predict_win([r1, r2])
    print(f"OpenSkill prediction for {team_1} vs {team_2}: {op:.2f}")

    # Elo final rating chart
    # plot_final_ratings(elo_ratings, "Top 10 Final Elo Ratings", name_map)

    # # TrueSkill mu chart
    # plot_final_ratings(
    #     trueskill_ratings,
    #     "Top 10 Final TrueSkill Ratings",
    #     name_map,
    #     getter=lambda r: r.mu - 3 * r.sigma,
    # )

    # # OpenSkill mu chart
    # plot_final_ratings(
    #     openskill_ratings,
    #     "Top 10 Final OpenSkill Ratings",
    #     name_map,
    #     getter=lambda r: r.mu - 3 * r.sigma,
    # )
    plot_roc_auc(y_true, elo_probs, ts_probs, os_probs)
    # plot_percentage_accuracy(elo_perc_count, elo_perc_sum, len(games), "elo")
    # plot_percentage_accuracy(ts_perc_count, ts_perc_sum, len(games), "ts")
    # plot_percentage_accuracy(os_perc_count, os_perc_sum, len(games), "os")
    plt.show()

    # mean = stat.mean(elo_diffs)
    # std = stat.stdev(elo_diffs)
    # # estymacja sigma pełnego rozkładu
    # sigma_hat = mean * math.sqrt(math.pi / 2)

    # # teoretyczne parametry half-normal
    # mean_theoretical = sigma_hat * math.sqrt(2 / math.pi)
    # std_theoretical = sigma_hat * math.sqrt(1 - 2 / math.pi)

    # print("Estimated full-normal σ:", sigma_hat)
    # print("Half-normal theoretical mean:", mean_theoretical)
    # print("Half-normal theoretical std:", std_theoretical)
    # plot_rating_diff_histogram(
    #     elo_diffs,
    #     "Elo Rating Difference Histogram",
    #     "elo_rating_diff",
    # )
    # plot_rating_diff_histogram(
    #     ts_diffs,
    #     "TrueSkill Mu Difference Histogram",
    #     "trueskill_diff",
    # )
    # plot_rating_diff_histogram(
    #     os_diffs,
    #     "OpenSkill Mu Difference Histogram",
    #     "openskill_diff",
    # )
    # plt.show()

    save_ratings(name_map, player_game_count)


if __name__ == "__main__":
    main()
