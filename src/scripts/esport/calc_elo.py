import json
from pathlib import Path
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, auc

import trueskill
from trueskill import Rating
from openskill.models import PlackettLuce

# Ranking update utilities
from utils.rankings._elo import update_team_elo, expected_win_elo
from utils.rankings._ts import update_trueskill, expected_trueskill_win, team_rating
from utils.rankings._os import update_openskill
from scipy.stats import randint, uniform

trueskill.DRAW_PROBABILITY = 0.0
elo_ratings = defaultdict(lambda: 1500)
ts_ratings = defaultdict(lambda: Rating())
os_model = PlackettLuce()
os_ratings = defaultdict(lambda: os_model.rating())

IDX_TO_ROLE = {
    0: "TOP",
    1: "JUNGLE",
    2: "MID",
    3: "ADC",
    4: "SUPPORT",
}


def predict_match_result(blue_ids, red_ids):
    """
    Given two lists of 5 player IDs each (blue_ids, red_ids), compute:
      - ELO-based win probability
      - TrueSkill-based win probability
      - OpenSkill-based win probability
    Then pick the team with the highest of these three probabilities as the “predicted winner.”

    Returns a dict with keys:
      {
        "elo_prob": p_blue_elo,
        "ts_prob": p_blue_ts,
        "os_prob": p_blue_os,
        "combined_predict": "blue" or "red"
      }
    """
    # 1) ELO probability
    avg_elo_blue = np.mean([elo_ratings[player_id] for player_id in blue_ids])
    avg_elo_red = np.mean([elo_ratings[player_id] for player_id in red_ids])
    # expected_win_elo(p1, p2) = probability that team1 beats team2
    p_blue_elo = expected_win_elo(avg_elo_blue, avg_elo_red)

    # 2) TrueSkill probability
    ts_ratings_blue = [ts_ratings[player_id] for player_id in blue_ids]
    ts_ratings_red = [ts_ratings[player_id] for player_id in red_ids]
    p_blue_ts = expected_trueskill_win(ts_ratings_blue, ts_ratings_red)

    # 3) OpenSkill probability
    os_ratings_blue = [os_ratings[player_id] for player_id in blue_ids]
    os_ratings_red = [os_ratings[player_id] for player_id in red_ids]
    p_blue_os, _ = os_model.predict_win([os_ratings_blue, os_ratings_red])

    # Decide which probability is highest
    probs = {"blue_elo": p_blue_elo, "blue_ts": p_blue_ts, "blue_os": p_blue_os}
    # Find the rating system that gives the largest probability for “blue” to win
    best_system = max(probs, key=lambda k: probs[k])
    # If that best prob > 0.5, predict blue; otherwise predict red.
    if probs[best_system] > 0.5:
        winner = "blue"
    else:
        winner = "red"

    return {
        "elo_prob": p_blue_elo,
        "ts_prob": p_blue_ts,
        "os_prob": p_blue_os,
        "combined_predict": winner,
    }


# Plotting utility for ROC curves
def plot_roc_auc(y_true, xgb_probs, elo_probs, ts_probs, os_probs):
    plt.figure(figsize=(6, 6))
    models = [
        # (rf_probs, "RandomForest"),
        (xgb_probs, "XGBoost"),
        (elo_probs, "ELO"),
        (ts_probs, "TrueSkill"),
        (os_probs, "OpenSkill"),
    ]
    for probs, label in models:
        print(f"Plotting {label} ROC curve...")
        fpr, tpr, _ = roc_curve(y_true, probs)
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {model_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def process_single_game(game):
    """
    Compute pre-game win probabilities for ELO, TrueSkill, OpenSkill,
    then update each rating system.
    Returns (p_e, p_t, p_o).
    """
    global elo_ratings, ts_ratings, os_model, os_ratings
    rankings = {}
    # Extract player IDs
    t1_ids = [p["player_id"] for p in game["t1_players"].values()]
    t2_ids = [p["player_id"] for p in game["t2_players"].values()]

    # ELO probability
    elo_ratings1 = [elo_ratings[p] for p in t1_ids]
    elo_ratings2 = [elo_ratings[p] for p in t2_ids]

    avg1 = np.mean(elo_ratings1)
    avg2 = np.mean(elo_ratings2)

    rankings["elo"] = (avg1, avg2)

    p_e = expected_win_elo(avg1, avg2)

    ts_ratings1 = [ts_ratings[p] for p in t1_ids]
    ts_ratings2 = [ts_ratings[p] for p in t2_ids]
    t1_ts_rank = team_rating(ts_ratings1)
    t2_ts_rank = team_rating(ts_ratings2)
    rankings["trueskill"] = (t1_ts_rank, t2_ts_rank)
    # TrueSkill probability
    p_t = expected_trueskill_win(
        ts_ratings1,
        ts_ratings2,
    )

    # OpenSkill probability
    os_ratings1 = [os_ratings[p] for p in t1_ids]
    os_ratings2 = [os_ratings[p] for p in t2_ids]
    t1_os_rank, t2_os_rank = os_model._calculate_team_ratings(
        [os_ratings1, os_ratings2]
    )
    rankings["openskill"] = (t1_os_rank, t2_os_rank)
    p_o, _ = os_model.predict_win([os_ratings1, os_ratings2])

    return (p_e, p_t, p_o), rankings


def update_rankings(game):
    # Update ratings based on result
    global elo_ratings, ts_ratings, os_model, os_ratings
    rankings = {}
    # Extract player IDs
    t1_ids = [p["player_id"] for p in game["t1_players"].values()]
    t2_ids = [p["player_id"] for p in game["t2_players"].values()]

    result = game.get("t1_win", False)

    elo_ratings1 = [elo_ratings[p] for p in t1_ids]
    elo_ratings2 = [elo_ratings[p] for p in t2_ids]

    # ELO update
    new1, new2 = update_team_elo(elo_ratings1, elo_ratings2, result, k=32)
    for pid, nr in zip(t1_ids + t2_ids, new1 + new2):
        elo_ratings[pid] = nr

    ts_ratings1 = [ts_ratings[p] for p in t1_ids]
    ts_ratings2 = [ts_ratings[p] for p in t2_ids]
    # TrueSkill update
    nt1, nt2 = update_trueskill(
        ts_ratings1,
        ts_ratings2,
        result,
    )
    for pid, nr in zip(t1_ids + t2_ids, nt1 + nt2):
        ts_ratings[pid] = nr

    os_ratings1 = [os_ratings[p] for p in t1_ids]
    os_ratings2 = [os_ratings[p] for p in t2_ids]
    # OpenSkill update
    o1, o2 = update_openskill(
        os_ratings1,
        os_ratings2,
        result,
        model=os_model,
    )
    for pid, nr in zip(t1_ids + t2_ids, o1 + o2):
        os_ratings[pid] = nr


HISTORY_SIZE = 50
team_hist = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
player_hist = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))


def get_team_stats(game, features):
    team_keys = [
        "kills",
        "towers",
        "dragons",
        "nashors",
        "gold",
        "game_duration",
        "enemy_rank",
        "won",
    ]
    win = int(game["t1_win"])
    hist = team_hist[game["t1_id"]]
    for k in team_keys:
        vals = [float(e[k]) for e in hist if e[k] is not None]

        if not vals:
            vals = [0.0]

        features[f"t1_{k}_avg"] = float(np.mean(vals)) if vals else 0.0

    hist = team_hist[game["t2_id"]]
    for k in team_keys:
        vals = [float(e[k]) for e in hist if e[k] is not None]

        if not vals:
            vals = [0.0]

        features[f"t2_{k}_avg"] = float(np.mean(vals)) if vals else 0.0

    t1_stats = game["t1_stats"]
    t2_stats = game["t2_stats"]
    t1_stats["game_duration"] = game["game_duration"]
    t2_stats["game_duration"] = game["game_duration"]
    t1_stats["enemy_rank"] = features["t2_elo"]
    t2_stats["enemy_rank"] = features["t1_elo"]
    t1_stats["won"] = win
    t2_stats["won"] = 1 - win
    return t1_stats, t2_stats


def get_player_stats(game, features):
    player_keys = [
        "level",
        "kills",
        "deaths",
        "assists",
        "cs",
        "golds",
        "gold%",
        "total_damage_to_champion",
        "gd@15",
        "csd@15",
        "xpd@15",
        "lvld@15",
        "elo",
    ]

    t1_players = list(game["t1_players"].values())
    t2_players = list(game["t2_players"].values())
    all_players = t1_players + t2_players

    pstats = {}

    for i, player in enumerate(all_players):
        hist = player_hist[player["player_id"]]
        rank = elo_ratings[player["player_id"]]
        stats = player["stats"]
        stats["elo"] = rank
        features[f"{i}_elo"] = rank
        for k in player_keys:

            vals = [float(e[k]) for e in hist if e[k] is not None]
            if not vals:
                vals = [0.0]
            val = float(np.mean(vals)) if vals else 0.0

            features[f"{(i // 5) + 1}_{IDX_TO_ROLE[i%5]}_{k}_avg"] = val

        pstats[player["player_id"]] = stats

    return pstats


def process_games(games):
    """
    Single-pass: updates ratings and builds features using fixed-position
    player features and team stats, without manual KDA handling.
    Returns y_true, elo_probs, ts_probs, os_probs, df.
    """

    elo_probs, ts_probs, os_probs = [], [], []

    records, y_true = [], []
    for game in tqdm(games, desc="Processing games"):
        (p_e, p_t, p_o), rankings = process_single_game(game)
        elo_probs.append(p_e)
        ts_probs.append(p_t)
        os_probs.append(p_o)

        win = int(game["t1_win"])
        y_true.append(win)

        t1_elo, t2_elo = rankings["elo"]
        t1_ts_rank, t2_ts_rank = rankings["trueskill"]
        t1_os_rank, t2_os_rank = rankings["openskill"]
        feat = {
            "date": game["date"],
            "elo_prob": p_e,
            "ts_prob": p_t,
            "os_prob": p_o,
            "t1_elo": t1_elo,
            "t2_elo": t2_elo,
            # "t1_ts_mu": t1_ts_rank.mu,
            # "t1_ts_sigma": t1_ts_rank.sigma,
            # "t2_ts_mu": t2_ts_rank.mu,
            # "t2_ts_sigma": t2_ts_rank.sigma,
            # "t1_os_mu": t1_os_rank.mu,
            # "t1_os_sigma": np.sqrt(t1_os_rank.sigma_squared),
            # "t2_os_mu": t2_os_rank.mu,
            # "t2_os_sigma": np.sqrt(t2_os_rank.sigma_squared),
            "team1_win": win,
        }

        pstats = get_player_stats(game, feat)
        # Team stats
        t1_stats, t2_stats = get_team_stats(game, feat)
        # Rolling player stats by position

        records.append(feat)

        # Update team and player histories
        team_hist[game["t1_id"]].append(t1_stats)
        team_hist[game["t2_id"]].append(t2_stats)

        for pid, player in pstats.items():
            player_hist[pid].append(pstats[pid])

        update_rankings(game)

    df = pd.DataFrame(records)
    return np.array(y_true), elo_probs, ts_probs, os_probs, df


def get_p_name_to_id_map(games) -> dict:
    """
    Create a mapping from player names to their IDs.
    This is useful for debugging and understanding the data.
    """
    p_name_to_id = {}
    for game in tqdm(games, desc="Building player name to ID map"):
        for team in ["t1_players", "t2_players"]:
            for player in game[team].values():
                p_name_to_id[player["player_name"]] = player["player_id"]
    return p_name_to_id


GLOBAL_XGB = None
GLOBAL_SCALER = None
GLOBAL_FEATURE_NAMES = None


def predict_with_xgb(blue_ids, red_ids):
    """
    Given two lists of 5 player IDs each (blue_ids, red_ids), this function:
      1) Computes ELO, TrueSkill, OpenSkill win‐probabilities (blue vs. red),
         plus avg ELO per team (t1_elo, t2_elo).
      2) Builds a one‐row pd.DataFrame whose columns match GLOBAL_FEATURE_NAMES.
         All columns not explicitly set below are filled with 0.0.
      3) Applies GLOBAL_SCALER.transform(...) to that DataFrame.
      4) Returns XGB’s predicted probability that “blue” wins (a float in [0,1]).
    """
    # 1) Compute “rating‐system” features exactly as in process_single_game
    # —————————————————————————
    # ELO
    avg_elo_blue = np.mean([elo_ratings[pid] for pid in blue_ids])
    avg_elo_red = np.mean([elo_ratings[pid] for pid in red_ids])
    p_blue_elo = expected_win_elo(avg_elo_blue, avg_elo_red)

    # TrueSkill
    ts_ratings_blue = [ts_ratings[pid] for pid in blue_ids]
    ts_ratings_red = [ts_ratings[pid] for pid in red_ids]
    p_blue_ts = expected_trueskill_win(ts_ratings_blue, ts_ratings_red)

    # OpenSkill
    os_ratings_blue = [os_ratings[pid] for pid in blue_ids]
    os_ratings_red = [os_ratings[pid] for pid in red_ids]
    p_blue_os, _ = os_model.predict_win([os_ratings_blue, os_ratings_red])

    # At this point we have:
    #   elo_prob = p_blue_elo
    #   ts_prob  = p_blue_ts
    #   os_prob  = p_blue_os
    #   t1_elo   = avg_elo_blue
    #   t2_elo   = avg_elo_red
    # —————————————————————————

    # 2) Build a one‐row dictionary, then turn into DataFrame with exactly GLOBAL_FEATURE_NAMES columns
    one_row = {col: 0.0 for col in GLOBAL_FEATURE_NAMES}

    # Fill in the five “rating‐based” columns:
    one_row["elo_prob"] = float(p_blue_elo)
    one_row["ts_prob"] = float(p_blue_ts)
    one_row["os_prob"] = float(p_blue_os)

    one_row["t1_elo"] = float(avg_elo_blue)
    one_row["t2_elo"] = float(avg_elo_red)

    # (everything else—team rolling‐stats or player rolling‐stats—stays 0.0)
    df_single = pd.DataFrame([one_row], columns=GLOBAL_FEATURE_NAMES)

    # 3) Scale
    X_scaled = GLOBAL_SCALER.transform(df_single)

    # 4) Predict with XGBoost
    prob_blue = GLOBAL_XGB.predict_proba(X_scaled)[0, 1]
    return prob_blue


def build_player_name_map(games):
    """
    Given a list of game‐dicts (the same structure as your data/games.json),
    return a dict: { player_id: player_name }.

    If a single player appears under multiple names over time, this will store
    whichever name appears last in the `games` list. If you want the first‐seen
    name, just check `if pid not in player_name_map` before assigning.
    """
    player_name_map = {}

    for game in games:
        # Each game has "t1_players" and "t2_players" as dicts of 5 player‐records
        for side in ("t1_players", "t2_players"):
            for p in game[side].values():
                pid = p["player_id"]
                name = p.get("player_name", "").strip()
                if name:
                    # Overwrite with the most recent name seen in the list order
                    player_name_map[pid] = name

    return player_name_map


def export_all_player_ratings(output_path="player_ratings.csv"):
    """
    Builds a DataFrame with one row per player‐ID, containing:
      pid, player_name, last_date, elo_rating, ts_mu, ts_sigma, os_mu, os_sigma
    from the globals:
      elo_ratings, ts_ratings, os_ratings, player_last_date, player_name_map
    """
    rows = []
    # Collect every pid seen in either ratings‐dict or last_date‐dict:
    all_pids = set(elo_ratings.keys()) | set(ts_ratings.keys()) | set(os_ratings.keys())

    for pid in all_pids:
        # 1) pid and player_name
        name = player_name_map.get(pid, "")

        # 2) last_date (could be empty string if they never appeared in a “game”)
        # last_dt = player_last_date.get(pid, "")

        # 3) ELO       (default 1500 if never rated)
        elo_value = elo_ratings.get(pid, 1500.0)

        # 4) TrueSkill → extract mu & sigma
        ts_r = ts_ratings.get(pid, trueskill.Rating())
        ts_mu = ts_r.mu
        ts_sigma = ts_r.sigma

        # 5) OpenSkill → extract mu & sigma (or sqrt(sigma_squared))
        os_r = os_ratings.get(pid, os_model.rating())
        os_mu = os_r.mu
        os_sigma = getattr(os_r, "sigma", None)
        if os_sigma is None:
            # fallback if only sigma_squared is defined
            ssq = getattr(os_r, "sigma_squared", None)
            os_sigma = (ssq**0.5) if (ssq is not None) else None

        rows.append(
            {
                "pid": pid,
                "player_name": name,
                # "last_date": last_dt,
                "elo_rating": elo_value,
                "ts_mu": ts_mu,
                "ts_sigma": ts_sigma,
                "os_mu": os_mu,
                "os_sigma": os_sigma,
            }
        )

    df = pd.DataFrame(rows)
    # (Optional) sort however you like, e.g. by last_date or pid
    df = df.sort_values("pid")
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} players to '{output_path}'")


player_name_map = {}


def train_models_from_process(y_train, X_train, y_test, X_test, random_state=42):
    """
    Trains LogisticRegression i XGBoost na podanych danych.
    """
    # Skalowanie
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    logistic = LogisticRegression(random_state=random_state)
    logistic.fit(X_train_scaled, y_train)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        learning_rate=0.03,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
    )
    xgb.fit(X_train_scaled, y_train)

    return logistic, xgb, scaler


if __name__ == "__main__":
    # Wczytaj dane
    data_path = Path("data/games.json")
    if not data_path.exists():
        raise FileNotFoundError("data/games.json not found")
    with open(data_path) as f:
        games = json.load(f)

    # Przetworzenie gier na cechy i prawdziwe wyniki
    y_true, elo_probs, ts_probs, os_probs, df = process_games(games)
    print(f"Processed {len(df)} games, {len(df.columns)} features.")

    # Przygotowanie macierzy cech
    X = df.drop(columns=["date", "team1_win"]).fillna(0.0).values
    y = np.array(y_true)
    elo_arr = np.array(elo_probs)
    ts_arr = np.array(ts_probs)
    os_arr = np.array(os_probs)

    # Split ze wspólnym indeksem
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Trenuj modele na cechach
    log_reg, xgb_clf, scaler = train_models_from_process(
        y_train, X_train, y_test, X_test
    )

    # Ewaluacja wszystkich metod
    methods = {
        "LogisticRegression": log_reg.predict_proba(scaler.transform(X_test))[:, 1],
        "XGBoost": xgb_clf.predict_proba(scaler.transform(X_test))[:, 1],
        "ELO": elo_arr[test_idx],
        "TrueSkill": ts_arr[test_idx],
        "OpenSkill": os_arr[test_idx],
    }

    for name, probs in methods.items():
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        recall = recall_score(y_test, preds)
        auc_score = roc_auc_score(y_test, probs)
        print(
            f"{name}: Accuracy = {acc:.2%}, Recall = {recall:.2%}, AUC = {auc_score:.3f}"
        )

    # Plot ROC curves
    plt.figure(figsize=(8, 8))
    for probs, label in [(methods[m], m) for m in methods]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
