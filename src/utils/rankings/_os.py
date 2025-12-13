from openskill.models import PlackettLuce, PlackettLuceRating


def update_openskill(
    team1: list[PlackettLuceRating],
    team2: list[PlackettLuceRating],
    t1_win: bool,
    model: PlackettLuce,
) -> tuple[list[PlackettLuceRating], list[PlackettLuceRating]]:
    teams = [team1, team2] if t1_win else [team2, team1]
    updated = model.rate(teams)
    return updated if t1_win else updated[::-1]


def expected_openskill_win(
    team1: list[PlackettLuceRating],
    team2: list[PlackettLuceRating],
    model: PlackettLuce,
) -> float:
    prob = model.predict_win([team1, team2])
    return prob[0]
