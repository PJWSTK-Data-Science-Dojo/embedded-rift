from math import sqrt
from trueskill import BETA, Rating, rate
from trueskill.backends import cdf


def team_rating(team):
    team_mu = sum(player.mu for player in team)
    team_sigma_squared = sum(player.sigma**2 for player in team)
    return team_mu, team_sigma_squared


def expected_trueskill_win(team1, team2):
    team1_mu, team1_sigma_squared = team_rating(team1)
    team2_mu, team2_sigma_squared = team_rating(team2)
    delta_mu = team1_mu - team2_mu
    denom = sqrt(2 * (BETA**2) + team1_sigma_squared + team2_sigma_squared)
    return cdf(delta_mu / denom)


def update_trueskill(team1: list[Rating], team2: list[Rating], t1_win):
    ranked = [team1, team2] if t1_win else [team2, team1]
    new_ratings = rate(ranked)

    t1 = new_ratings[0] if t1_win else new_ratings[1]
    t2 = new_ratings[1] if t1_win else new_ratings[0]

    return t1, t2
