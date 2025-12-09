from math import sqrt


def expected_win_elo(r1, r2) -> float:
    return 1 / (1 + 10 ** ((r2 - r1) / (400)))


def expected_win_elo_team(team1, team2) -> float:
    """
    Calculate the expected win probability for team1 against team2.
    Uses the average Elo rating of each team.
    """
    t1_avg = sum(team1) / len(team1)
    t2_avg = sum(team2) / len(team2)
    return expected_win_elo(t1_avg, t2_avg)


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
        p_up, _ = update_elo(pi, t2_avg, win, k=k)
        t1_up.append(pi + p_up)

    t2_up = []
    for pi in team2:
        p_up, _ = update_elo(pi, t1_avg, 1 - win, k=k)
        t2_up.append(pi + p_up)

    return t1_up, t2_up
