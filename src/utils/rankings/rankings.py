import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

import trueskill
from trueskill import Rating
from openskill.models import PlackettLuce
from choix import ilsr_pairwise

from utils.rankings._elo import update_team_elo, expected_win_elo
from utils.rankings._ts import update_trueskill, expected_trueskill_win, team_rating
from utils.rankings._os import update_openskill

trueskill.DRAW_PROBABILITY = 0.0

elo_ratings = defaultdict(lambda: 1500.0)
ts_ratings = defaultdict(lambda: Rating())
os_model = PlackettLuce()
os_ratings = defaultdict(lambda: os_model.rating())

# team_id -> index
bt_team_index = {}
# store (winner_idx, loser_idx, date_str "YYYY-MM-DD")
bt_results = []
bt_strengths = np.zeros(0)


def reset_all_rankings():
    global elo_ratings, ts_ratings, os_model, os_ratings
    global bt_team_index, bt_results, bt_strengths

    elo_ratings = defaultdict(lambda: 1500.0)
    ts_ratings = defaultdict(lambda: Rating())
    os_model = PlackettLuce()
    os_ratings = defaultdict(lambda: os_model.rating())

    bt_team_index = {}
    bt_results = []
    bt_strengths = np.zeros(0)


def _ensure_bt_team_idx(team_id):
    global bt_team_index, bt_strengths
    if team_id not in bt_team_index:
        idx = len(bt_team_index)
        bt_team_index[team_id] = idx
        if bt_strengths.size == 0:
            bt_strengths = np.zeros(1)
        else:
            bt_strengths = np.concatenate([bt_strengths, [0.0]])
    return bt_team_index[team_id]


def update_bradley_terry(game, alpha=0.01, max_iter=100):
    """
    Uses only games from the last 2 years (relative to current game['date']).
    """
    from math import isfinite

    global bt_results, bt_strengths

    t1 = _ensure_bt_team_idx(game["t1_id"])
    t2 = _ensure_bt_team_idx(game["t2_id"])
    winner = t1 if game["t1_win"] else t2
    loser = t2 if game["t1_win"] else t1

    date_str = game["date"]  # "YYYY-MM-DD"
    bt_results.append((winner, loser, date_str))

    if len(bt_team_index) == 0:
        return

    current_dt = datetime.strptime(date_str, "%Y-%m-%d")
    cutoff_dt = current_dt - timedelta(days=365 * 2)

    filtered_pairs = [
        (w, l)
        for (w, l, d) in bt_results
        if datetime.strptime(d, "%Y-%m-%d") >= cutoff_dt
    ]

    if not filtered_pairs:
        return

    n_items = len(bt_team_index)
    bt_strengths = ilsr_pairwise(
        n_items=n_items,
        data=filtered_pairs,
        alpha=alpha,
        max_iter=max_iter,
    )

    bt_strengths = np.array([s if isfinite(s) else 0.0 for s in bt_strengths])


def bradley_terry_win_prob(team1_id, team2_id):
    if len(bt_team_index) == 0:
        return 0.5

    i1 = _ensure_bt_team_idx(team1_id)
    i2 = _ensure_bt_team_idx(team2_id)

    s1 = bt_strengths[i1] if i1 < len(bt_strengths) else 0.0
    s2 = bt_strengths[i2] if i2 < len(bt_strengths) else 0.0
    diff = s1 - s2
    return float(1.0 / (1.0 + np.exp(-diff)))
