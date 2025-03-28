import csv
import math
from tqdm import tqdm
from utils.champion import CHAMPION_IDS
from utils.hdf5 import HDF5Database

import trueskill
# For Elo we use our simple update function (unchanged)
def update_elo(champion_ratings, blue, red, blue_win, K=32):
    if blue_win:
        avg_red = sum(champion_ratings[c] for c in red) / len(red)
        avg_blue = sum(champion_ratings[c] for c in blue) / len(blue)
        for champ in blue:
            r_old = champion_ratings[champ]
            expected = 1 / (1 + 10 ** ((avg_red - r_old) / 400))
            champion_ratings[champ] = r_old + K * (1 - expected)
        for champ in red:
            r_old = champion_ratings[champ]
            expected = 1 / (1 + 10 ** ((avg_blue - r_old) / 400))
            champion_ratings[champ] = r_old + K * (0 - expected)
    else:
        avg_blue = sum(champion_ratings[c] for c in blue) / len(blue)
        avg_red = sum(champion_ratings[c] for c in red) / len(red)
        for champ in blue:
            r_old = champion_ratings[champ]
            expected = 1 / (1 + 10 ** ((avg_red - r_old) / 400))
            champion_ratings[champ] = r_old + K * (0 - expected)
        for champ in red:
            r_old = champion_ratings[champ]
            expected = 1 / (1 + 10 ** ((avg_blue - r_old) / 400))
            champion_ratings[champ] = r_old + K * (1 - expected)

# Initialize global rating dictionaries.
champion_ts = {champ: trueskill.Rating() for champ in CHAMPION_IDS}
champion_elo = {champ: 1500 for champ in CHAMPION_IDS}

# --- Updated OpenSkill Part ---
# Instead of using a raw os_rate function, we now import the PlackettLuce model
from openskill.models import PlackettLuce
# Create a global OpenSkill model instance with default parameters.
os_model = PlackettLuce()
# Initialize champion ratings for OpenSkill using its model's rating() function.
champion_os = {champ: os_model.rating() for champ in CHAMPION_IDS}

def update_trueskill(champion_ratings, blue, red, blue_win):
    blue_team = [champion_ratings[c] for c in blue]
    red_team = [champion_ratings[c] for c in red]
    if blue_win:
        new_blue, new_red = trueskill.rate([blue_team, red_team], ranks=[0, 1])
    else:
        new_blue, new_red = trueskill.rate([blue_team, red_team], ranks=[1, 0])
    for i, champ in enumerate(blue):
        champion_ratings[champ] = new_blue[i]
    for i, champ in enumerate(red):
        champion_ratings[champ] = new_red[i]

# Updated OpenSkill update function
def update_openskill(champion_ratings, blue, red, blue_win):
    blue_team = [champion_ratings[c] for c in blue]
    red_team = [champion_ratings[c] for c in red]
    if blue_win:
        new_blue, new_red = os_model.rate([blue_team, red_team], ranks=[0, 1])
    else:
        new_blue, new_red = os_model.rate([blue_team, red_team], ranks=[1, 0])
    for i, champ in enumerate(blue):
        champion_ratings[champ] = new_blue[i]
    for i, champ in enumerate(red):
        champion_ratings[champ] = new_red[i]

def predict_game_ts(champion_ratings, blue, red):
    blue_mu = sum(champion_ratings[c].mu for c in blue)
    red_mu = sum(champion_ratings[c].mu for c in red)
    beta = trueskill.global_env().beta
    blue_sigma2 = sum(champion_ratings[c].sigma ** 2 for c in blue)
    red_sigma2 = sum(champion_ratings[c].sigma ** 2 for c in red)
    denom = math.sqrt(len(blue) * (beta ** 2) + blue_sigma2 + red_sigma2)
    prob_blue = trueskill.global_env().cdf((blue_mu - red_mu) / denom)
    return prob_blue  # probability that blue wins

def predict_game_elo(champion_ratings, blue, red):
    avg_blue = sum(champion_ratings[c] for c in blue) / len(blue)
    avg_red = sum(champion_ratings[c] for c in red) / len(red)
    return 1 / (1 + 10 ** ((avg_red - avg_blue) / 400))

def save_ratings(filename, ratings_dict, system_name):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ChampionID", f"{system_name}_Rating"])
        for champ, rating in ratings_dict.items():
            if isinstance(rating, (trueskill.Rating)) or hasattr(rating, "mu"):
                writer.writerow([champ, f"mu={rating.mu:.3f}, sigma={rating.sigma:.3f}"])
            else:
                writer.writerow([champ, rating])

def main(db: HDF5Database):
    ids = db.get_game_ids()
    ids = sorted(ids, key=lambda x: int(x.split("_")[-1]))
    
    split_index = int(0.9 * len(ids))
    train_ids = ids[:split_index]
    test_ids = ids[split_index:]
    
    # Training loop with progress bar.
    for gid in tqdm(train_ids, desc="Training Games"):
        game_data = db.get_game(gid)
        champions_played = db.get_champions(gid)
        blue = champions_played[0, :]  # champion IDs for blue team.
        red = champions_played[1, :]   # champion IDs for red team.
        blue_win = game_data.attrs["blue_win"]

        update_trueskill(champion_ts, blue, red, blue_win)
        update_openskill(champion_os, blue, red, blue_win)
        update_elo(champion_elo, blue, red, blue_win)
    
    ts_correct = 0
    os_correct = 0
    elo_correct = 0
    total = 0
    brier_elo_total = 0.0
    brier_ts_total = 0.0
    brier_os_total = 0.0
    # Testing loop with progress bar.
    for gid in tqdm(test_ids, desc="Testing Games"):
        game_data = db.get_game(gid)
        champions_played = db.get_champions(gid)
        blue = champions_played[0, :]
        red = champions_played[1, :]
        bwin = game_data.attrs["blue_win"]
        total += 1

        # Elo prediction.
        elo_prob_blue = predict_game_elo(champion_elo, blue, red)
        
        if elo_prob_blue >= 0.5 and bwin:
            elo_correct += 1
        elif elo_prob_blue < 0.5 and not bwin:
            elo_correct += 1
        brier_elo_total += (elo_prob_blue - bwin) ** 2
        # OpenSkill prediction.
        # Using os_model.predict_win function if available; otherwise, use a similar approach:
        os_preds = os_model.predict_win([[champion_os[c] for c in blue],
                                         [champion_os[c] for c in red]])
        if os_preds[0] >= os_preds[1] and bwin:
            os_correct += 1
        elif os_preds[0] < os_preds[1] and not bwin:
            os_correct += 1
        brier_os_total += (os_preds[0] - bwin) ** 2
        # TrueSkill prediction.
        ts_prob_blue = predict_game_ts(champion_ts, blue, red)
        if ts_prob_blue >= 0.5 and bwin:
            ts_correct += 1
        elif ts_prob_blue < 0.5 and not bwin:
            ts_correct += 1
        brier_ts_total += (ts_prob_blue - bwin) ** 2
        
    ts_accuracy = ts_correct / total * 100
    os_accuracy = os_correct / total * 100
    elo_accuracy = elo_correct / total * 100

    brier_elo = brier_elo_total / total
    brier_ts = brier_ts_total / total
    brier_os = brier_os_total / total
    
    print("Test set accuracy:")
    print(f"TrueSkill: {ts_accuracy:.2f}%")
    print(f"OpenSkill: {os_accuracy:.2f}%")
    print(f"Elo: {elo_accuracy:.2f}%")
    print("\nBrier Scores:")
    print(f"Elo Brier Score: {brier_elo:.4f}")
    print(f"TrueSkill Brier Score: {brier_ts:.4f}")
    print(f"OpenSkill Brier Score: {brier_os:.4f}")
    print("\nForecast skill:")
    fs_elo = 1 - (brier_elo / 0.25)
    fs_ts = 1 - (brier_ts / 0.25)
    fs_os = 1 - (brier_os / 0.25)
    print(f"Elo Forecast Skill: {fs_elo:.4f}")
    print(f"TrueSkill Forecast Skill: {fs_ts:.4f}")
    print(f"OpenSkill Forecast Skill: {fs_os:.4f}")    
    
    # Save final ratings.
    save_ratings("final_trueskill_ratings.csv", champion_ts, "TrueSkill")
    save_ratings("final_openskill_ratings.csv", champion_os, "OpenSkill")
    save_ratings("final_elo_ratings.csv", champion_elo, "Elo")

if __name__ == "__main__":
    with HDF5Database() as db:
        main(db)
