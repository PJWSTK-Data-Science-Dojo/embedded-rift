import json


def parse_match_result(match_result_json):
    parsed_result = {}
    metadata = {
        "matchId": match_result_json["metadata"]["matchId"],
        "participantsPUUID:": match_result_json["metadata"]["participants"],
    }

    season, patch, *_ = match_result_json["info"]["gameVersion"].split(".")
    metadata["season"] = season
    metadata["patch"] = patch
    metadata["platform"] = match_result_json["info"]["platformId"]

    parsed_result["metadata"] = metadata

    result = {
        "gameStartTimestamp": match_result_json["info"]["gameStartTimestamp"],
        "gameEndTimestamp": match_result_json["info"]["gameEndTimestamp"],
        "gameDuration": match_result_json["info"]["gameDuration"],
    }

    participants = []
    keys_to_remove = [
        "challenges",
        "missions",
    ]
    for participant in match_result_json["info"]["participants"]:
        for key in keys_to_remove:
            participant.pop(key, None)
        participants.append(participant)

    result["teams"] = {}
    result["teams"]["blue"] = match_result_json["info"]["teams"][0]
    result["teams"]["red"] = match_result_json["info"]["teams"][1]

    result["teams"]["blue"]["participants"], result["teams"]["red"]["participants"] = (
        participants[:5],
        participants[5:],
    )

    parsed_result["result"] = result

    return parsed_result
