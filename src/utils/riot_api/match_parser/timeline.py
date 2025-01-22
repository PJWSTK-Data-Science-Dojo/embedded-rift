from . import Team

IGNORED = {2003, 2031, 2055,  # Health Potion, Refillable Potion, Control Ward
           2138, 2139, 2140,  # Elixir of Iron, Elixir of Sorcery, Elixir of Wrath
           3340, 3363, 3364}  # Stealth Ward, Farsight Alteration, Oracle Lens

ELITE_MONSTERS = ['HORDE', 'RIFTHERALD', 'BARON_NASHOR', 'DRAGON']  # monster types used by RiotAPI
ELITE_MONSTERS_LABELS = ['voidGrub', 'riftHerald', 'baronNashor', 'drake', 'elderDrake']

SOULS = ['None', 'Infernal', 'Ocean', 'Cloud', 'Mountain', 'Hextech', 'Chemtech']  # possible dragon souls, same names used by RiotAPI

BUILDINGS = ['TOWER_BUILDING', 'INHIBITOR_BUILDING']  # building types used by RiotAPI
BUILDINGS_LABELS = ['turret', 'inhibitor']


def parse_timeline(timeline_json: dict) -> dict:
    frames = timeline_json['info']['frames']
    parsed_frames = []

    for i, frame in enumerate(frames):
        # Initialize frame dict
        frame_dict = {
            'teams': [
                {'eventData': init_team_event_data()} for _ in range(2)
            ],
            'participants': [
                {'eventData': init_player_event_data()} for _ in range(10)
            ]
        }

        if i > 0:
            # Load event data from previous frame
            for team_id in range(2):
                frame_dict['teams'][team_id]['eventData'] = parsed_frames[i-1]['teams'][team_id]['eventData'].copy()
            for participant_id in range(10):
                frame_dict['participants'][participant_id]['eventData'] = parsed_frames[i-1]['participants'][participant_id]['eventData'].copy()

        for event in frame['events']:
            # Update current frame dict based on happening events
            if event['type'] == 'CHAMPION_KILL':
                frame_dict = parse_champion_kill_event(frame_dict, event)
            elif event['type'] in ['ITEM_PURCHASED', 'ITEM_DESTROYED', 'ITEM_UNDO', 'ITEM_SOLD']:
                frame_dict = parse_item_event(frame_dict, event)
            elif event['type'] == 'SKILL_LEVEL_UP':
                frame_dict = parse_skill_level_up_event(frame_dict, event)
            elif event['type'] == 'ELITE_MONSTER_KILL':
                frame_dict = parse_elite_monster_kill_event(frame_dict, event)
            elif event['type'] == 'DRAGON_SOUL_GIVEN':
                frame_dict = parse_dragon_soul_given_event(frame_dict, event)
            elif event['type'] == 'BUILDING_KILL':
                frame_dict = parse_building_kill_event(frame_dict, event)
            elif event['type'] == 'TURRET_PLATE_DESTROYED':
                frame_dict = parse_turret_plate_destroyed_event(frame_dict, event)
            elif event['type'] in ['WARD_PLACED', 'WARD_KILL']:
                frame_dict = parse_ward_event(frame_dict, event)

        # Include non-event data like championStats, damageStats ...
        for player_id in range(10):
            participantFrame = frame['participantFrames'][str(player_id+1)]
            frame_dict['participants'][player_id] = {**frame_dict['participants'][player_id], **participantFrame}
        parsed_frames.append(frame_dict)

    for i in range(len(parsed_frames)):
        participants = parsed_frames[i].pop('participants', None)
        for team_id in range(2):
            parsed_frames[i]['teams'][team_id]['participants'] = participants[team_id:team_id+5]

    return parsed_frames

def init_team_event_data() -> dict[str, int | dict[str, int]]:
    event_data = {
        'eliteMonstersKilled': {label: 0 for label in ELITE_MONSTERS_LABELS},
        'dragonSoul': 0,
        'buildingsDestroyed': {label: 0 for label in BUILDINGS_LABELS}
    }

    return event_data


def init_player_event_data() -> dict[str, int | list[int]]:
    event_data = {
        'kills': 0, 'deaths': 0, 'assists': 0,
        'items': [0, 0, 0, 0, 0, 0],
        'skills': [0, 0, 0, 0],
        'turretPlatesDestroyed': 0,
        'wardsPlaced': 0, 'wardsDestroyed': 0
    }

    return event_data


def parse_champion_kill_event(frame_dict: dict, event: dict) -> dict:
    participants = frame_dict['participants']

    if 'killerId' in event.keys():
        killer_id = event['killerId'] - 1
        participants[killer_id]['eventData']['kills'] = participants[killer_id]['eventData'].setdefault('kills', 0) + 1

    victim_id = event['victimId'] - 1
    participants[victim_id]['eventData']['deaths'] = participants[victim_id]['eventData'].setdefault('deaths', 0) + 1
    assistant_ids = []
    if 'assistingParticipantIds' in event.keys():
        assistant_ids = [i - 1 for i in event['assistingParticipantIds']]

    for assistant_id in assistant_ids:
        participants[assistant_id]['eventData']['assists'] = participants[assistant_id]['eventData'].setdefault('assists', 0) + 1

    return frame_dict


def parse_item_event(frame_dict, event):
    player_id = event['participantId'] - 1
    items: list = frame_dict['participants'][player_id]['eventData']['items']

    if event['type'] == 'ITEM_PURCHASED':
        if event['itemId'] not in IGNORED:
            index = items.index(0)
            items[index] = event['itemId']

    elif event['type'] in ['ITEM_DESTROYED', 'ITEM_SOLD']:
        try:
            index = items.index(event['itemId'])
            items[index] = 0
        except ValueError:  # One of the ignored items or possibly an item received from runes
            pass

    elif event['type'] == 'ITEM_UNDO':
        if event['afterId'] not in IGNORED and event['beforeId'] not in IGNORED:
            # TODO: Check if this is correct
            try:
                index = items.index(event['beforeId'])
                items[index] = event['afterId']
            except ValueError:
                pass

    return frame_dict


def parse_skill_level_up_event(frame_dict, event):
    player_id = event['participantId'] - 1
    skill_index = event['skillSlot'] - 1
    skills = frame_dict['participants'][player_id]['eventData']['skills']
    skills[skill_index] += 1

    return frame_dict


def parse_elite_monster_kill_event(frame_dict: dict, event: dict) -> dict:
    team_id = 0 if event['killerTeamId'] == Team.BLUE else 1 # team_id - id of the team that killed the monster
    
    monster_id = ELITE_MONSTERS.index(event['monsterType'])
    
    if event['monsterType'] == 'DRAGON' and event['monsterSubType'] == 'ELDER_DRAGON':
        monster_id = ELITE_MONSTERS_LABELS.index('elderDrake')

    elite_monsters_killed = frame_dict['teams'][team_id]['eventData']['eliteMonstersKilled']
    elite_monsters_killed[ELITE_MONSTERS_LABELS[monster_id]] += 1

    return frame_dict


def parse_dragon_soul_given_event(frame_dict: dict, event: dict) -> dict:
    team_id = 0 if event['teamId'] == Team.BLUE else 1  # team_id - id of the team that gets the soul
    frame_dict['teams'][team_id]['eventData']['dragonSoul'] = SOULS.index(event['name'])

    return frame_dict


def parse_building_kill_event(frame_dict: dict, event: dict) -> dict:
    # teamId: (id of the team to which the building belonged)
    team_id = 0 if event['teamId'] == Team.RED else 1  # team_id - id of the team that destroyed the building

    if event['buildingType'] in BUILDINGS:
        building_id = BUILDINGS.index(event['buildingType'])  # Tower, Inhibitor
        frame_dict['teams'][team_id]['eventData']['buildingsDestroyed'][BUILDINGS_LABELS[building_id]] += 1

    return frame_dict


def parse_turret_plate_destroyed_event(frame_dict: dict, event: dict) -> dict:
    player_id = event['killerId']
    if player_id >= len(frame_dict['participants']):
        return frame_dict
    
    frame_dict['participants'][player_id]['eventData']['turretPlatesDestroyed'] += 1

    return frame_dict


def parse_ward_event(frame_dict: dict, event: dict) -> dict:
    if event['type'] == 'WARD_PLACED':
        player_id = event['creatorId'] - 1
        frame_dict['participants'][player_id]['eventData']['wardsPlaced'] += 1
        return frame_dict
    

    if event['type'] == 'WARD_KILL':
        player_id = event['killerId'] - 1
        frame_dict['participants'][player_id]['eventData']['wardsDestroyed'] += 1
        return frame_dict

    return frame_dict
