from math import sqrt


def expected_win_elo(r1, r2) -> float:
    return 1 / (1 + 10 ** ((r2 - r1) / (400)))


# For Elo we use our simple update function (unchanged)
def update_elo(p1, p2, p1_win, k=32):
    outcome = 1 if p1_win else 0

    p1_up = k * (outcome - expected_win_elo(p1, p2))
    p2_up = k * ((1 - outcome) - (expected_win_elo(p2, p1)))

    return p1_up, p2_up


def update_team_elo(team1, team2, t1_win: bool, k=64):
    t1_avg = sum(team1) / len(team1)
    t2_avg = sum(team2) / len(team2)
    win = 1 if t1_win else 0
    t1_up = []
    for pi in team1:
        p_up, _ = update_elo(pi, t2_avg, win)
        t1_up.append(pi + p_up)

    t2_up = []
    for pi in team2:
        p_up, _ = update_elo(pi, t1_avg, 1 - win)
        t2_up.append(pi + p_up)

    return t1_up, t2_up
