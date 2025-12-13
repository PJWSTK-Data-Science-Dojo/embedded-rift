import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import trueskill
from trueskill import Rating
from openskill.models import PlackettLuce
from choix import ilsr_pairwise

trueskill.DRAW_PROBABILITY = 0.0

from utils.rankings._elo import update_team_elo, expected_win_elo
from utils.rankings._ts import update_trueskill, expected_trueskill_win
from utils.rankings._os import update_openskill

team_elo = defaultdict(lambda: 1500.0)
team_ts = defaultdict(lambda: Rating())
os_model = PlackettLuce()
team_os = defaultdict(lambda: os_model.rating())

bt_team_index = {}
bt_pairs = []
BT_PARAMS = None


def get_bt_index(team_id):
    if team_id not in bt_team_index:
        bt_team_index[team_id] = len(bt_team_index)
    return bt_team_index[team_id]


def team_pre_match_probs(game):
    t1_id = game["t1_id"]
    t2_id = game["t2_id"]

    r1_elo = team_elo[t1_id]
    r2_elo = team_elo[t2_id]
    p_e = expected_win_elo(r1_elo, r2_elo)

    r1_ts = team_ts[t1_id]
    r2_ts = team_ts[t2_id]
    p_t = expected_trueskill_win([r1_ts], [r2_ts])

    r1_os = team_os[t1_id]
    r2_os = team_os[t2_id]
    p_o, _ = os_model.predict_win([[r1_os], [r2_os]])

    return {"elo": p_e, "ts": p_t, "os": p_o}


def update_team_ratings(game):
    global team_elo, team_ts, team_os, bt_pairs

    t1_id = game["t1_id"]
    t2_id = game["t2_id"]
    t1_win = int(game["t1_win"])

    r1_elo = team_elo[t1_id]
    r2_elo = team_elo[t2_id]
    new1_elo, new2_elo = update_team_elo([r1_elo], [r2_elo], t1_win, k=32)
    team_elo[t1_id] = new1_elo[0]
    team_elo[t2_id] = new2_elo[0]

    r1_ts = team_ts[t1_id]
    r2_ts = team_ts[t2_id]
    new1_ts, new2_ts = update_trueskill([r1_ts], [r2_ts], t1_win)
    team_ts[t1_id] = new1_ts[0]
    team_ts[t2_id] = new2_ts[0]

    r1_os = team_os[t1_id]
    r2_os = team_os[t2_id]
    new1_os, new2_os = update_openskill([r1_os], [r2_os], t1_win, model=os_model)
    team_os[t1_id] = new1_os[0]
    team_os[t2_id] = new2_os[0]

    i1 = get_bt_index(t1_id)
    i2 = get_bt_index(t2_id)
    if t1_win == 1:
        bt_pairs.append((i1, i2))
    else:
        bt_pairs.append((i2, i1))


def process_games_team_level(games):
    y_true = []
    elo_probs = []
    ts_probs = []
    os_probs = []
    records = []

    for g in tqdm(games, desc="Processing games (teams)"):
        t1_id = g["t1_id"]
        t2_id = g["t2_id"]
        win = int(g["t1_win"])

        probs = team_pre_match_probs(g)

        y_true.append(win)
        elo_probs.append(probs["elo"])
        ts_probs.append(probs["ts"])
        os_probs.append(probs["os"])

        records.append(
            {
                "date": g["date"],
                "t1_id": t1_id,
                "t2_id": t2_id,
                "t1_win": win,
                "elo_prob": probs["elo"],
                "ts_prob": probs["ts"],
                "os_prob": probs["os"],
            }
        )

        update_team_ratings(g)

    df = pd.DataFrame(records)
    return (
        np.array(y_true),
        np.array(elo_probs),
        np.array(ts_probs),
        np.array(os_probs),
        df,
    )


def fit_bradley_terry():
    global BT_PARAMS
    n_teams = len(bt_team_index)
    if n_teams == 0 or len(bt_pairs) == 0:
        BT_PARAMS = None
        return
    BT_PARAMS = ilsr_pairwise(n_teams, bt_pairs)


def bt_prob(team1_id, team2_id):
    if BT_PARAMS is None:
        raise RuntimeError("Bradley–Terry not fitted")
    i = get_bt_index(team1_id)
    j = get_bt_index(team2_id)
    d = BT_PARAMS[i] - BT_PARAMS[j]
    return 1.0 / (1.0 + np.exp(-d))


def export_team_rankings(output_path="team_rankings.csv"):
    rows = []
    all_ids = (
        set(team_elo.keys())
        | set(team_ts.keys())
        | set(team_os.keys())
        | set(bt_team_index.keys())
    )

    for tid in all_ids:
        elo_val = team_elo.get(tid, 1500.0)
        ts_r = team_ts.get(tid, Rating())
        os_r = team_os.get(tid, os_model.rating())

        ts_mu, ts_sigma = ts_r.mu, ts_r.sigma
        os_mu = os_r.mu
        os_sigma = getattr(os_r, "sigma", None)
        if os_sigma is None:
            ssq = getattr(os_r, "sigma_squared", None)
            os_sigma = (ssq**0.5) if ssq is not None else None

        bt_theta = None
        if BT_PARAMS is not None and tid in bt_team_index:
            bt_theta = BT_PARAMS[bt_team_index[tid]]

        rows.append(
            {
                "team_id": tid,
                "elo": elo_val,
                "ts_mu": ts_mu,
                "ts_sigma": ts_sigma,
                "os_mu": os_mu,
                "os_sigma": os_sigma,
                "bt_theta": bt_theta,
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values("elo", ascending=False, inplace=True)
    df.to_csv(output_path, index=False)


def predict_match_team_level(team1_id, team2_id):
    dummy = {"t1_id": team1_id, "t2_id": team2_id, "t1_win": 0}
    probs = team_pre_match_probs(dummy)
    bt_p = None
    if BT_PARAMS is not None:
        bt_p = bt_prob(team1_id, team2_id)
    return {
        "elo_prob": probs["elo"],
        "ts_prob": probs["ts"],
        "os_prob": probs["os"],
        "bt_prob": bt_p,
    }


if __name__ == "__main__":
    data_path = Path("data/games.json")
    if not data_path.exists():
        raise FileNotFoundError("data/games.json not found")

    with open(data_path) as f:
        games = json.load(f)

    start_date = "2020-01-01"
    games = [g for g in games if g["date"] >= start_date]

    y_true, elo_probs, ts_probs, os_probs, df_games = process_games_team_level(games)
    fit_bradley_terry()
    export_team_rankings("team_rankings.csv")
