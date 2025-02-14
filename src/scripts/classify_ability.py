from dotenv import load_dotenv
from pprint import pprint
import pandas as pd
import numpy as np
import json
from utils.ability2vec._classifier import PROMPT, classify_ability, PROPERTIES_DICT

load_dotenv()

TARGETING_TYPE = ['passive', 'auto', 'direction', 'location', 'unit', 'vector', 'varied']
COST_TYPE = ['none', 'mana', 'mana_per_second', 'energy', 'health', 'fury', 'grit']

DIM = len(PROPERTIES_DICT) + len(TARGETING_TYPE) + len(COST_TYPE)

with open("champions.json") as f:
    champions_dict = json.load(f)


def fetch_ability(champion_name: str, abilities_to_fetch: list[str] = None) -> list[dict]:
    abilities_to_fetch = ['Q', 'W', 'E', 'R', 'I', 'A'] if abilities_to_fetch is None else abilities_to_fetch
    champion = champions_dict[champion_name]
    output = []

    for ability in champion['abilities']:
        if champion['abilities'][ability]['skill'] in abilities_to_fetch:
            output.append(champion['abilities'][ability])

    return output


def get_ability_description(ability: dict) -> str:
    return ability['description']


def get_ability_context(ability: dict) -> str:
    context = []
    for tooltip in ability['tooltips']:
        if ability['tooltips'][tooltip].keys():
            context.append(f"\n\n".join(['### ' + value for value in ability['tooltips'][tooltip].values()]))
    return f"\n\n".join(context)


def get_ability_cost_type(ability: dict) -> str:
    ability_cost_type = ability['costtype'].casefold()
    cost_type = 'none'
    for possible_cost_type in COST_TYPE:
        if possible_cost_type in ability_cost_type:
            if 'mana' in ability_cost_type:
                cost_type = 'mana' if 'per second' not in ability_cost_type else 'mana_per_second'
            else:
                cost_type = possible_cost_type
            break

    return cost_type


def embed_ability_cost_type(ability_cost_type: str) -> np.ndarray:
    cost_type_embedded = np.zeros(len(COST_TYPE))
    cost_type_embedded[COST_TYPE.index(ability_cost_type)] = 1.0

    return cost_type_embedded


def get_ability_targeting(ability: dict) -> tuple:
    ability_targeting = ability['targeting'].casefold()
    targeting = []
    [targeting.append(targeting_type) for targeting_type in ability_targeting.split(' / ')]

    return tuple(targeting)


def embed_ability_targeting(ability_targeting: tuple) -> np.ndarray:
    targeting_embedded = np.zeros(len(TARGETING_TYPE))
    for targeting_type in ability_targeting:
        if targeting_type in TARGETING_TYPE:
            targeting_embedded[TARGETING_TYPE.index(targeting_type)] = 1.0

    return targeting_embedded


def ability_dict_to_vector(ability: dict) -> np.ndarray:
    vec = np.zeros(len(PROPERTIES_DICT))
    prompt_results = [classify_ability(get_ability_description(ability), prompt=prompt.value, context=get_ability_context(ability)) for prompt in PROMPT]
    true_keys = []
    for prompt_result in prompt_results:
        true_keys.extend([key for key, value in prompt_result.items() if value is True])
    key_list = list(PROPERTIES_DICT.keys())

    for key in true_keys:
        try:
            vec[key_list.index(key)] = 1
        except ValueError:
            print(f"Error: property {key} does not exist")

    targeting_vec = embed_ability_targeting(get_ability_targeting(ability))
    cost_type_vec = embed_ability_cost_type(get_ability_cost_type(ability))
    vec = np.concatenate([vec, targeting_vec, cost_type_vec])

    return vec


def generate_ability_dataframe() -> pd.DataFrame:
    ability_form_count, ability_vectors = {}, {}
    for i, champion_name in enumerate(champions_dict):
        print(f"Starting {champion_name}...")
        try:
            abilities = fetch_ability(champion_name)
            for ability in abilities:
                ability_name = f"{champion_name}_{ability['skill']}"
                ability_form_count[ability_name] = ability_form_count.setdefault(ability_name, 0) + 1
                ability_vectors[ability_name] = ability_vectors.setdefault(ability_name, np.zeros(DIM)) + ability_dict_to_vector(ability)

            for key in ability_vectors.keys():
                ability_vectors[key] = ability_vectors[key] / ability_form_count[key]

            columns = [*list(PROPERTIES_DICT.keys()), *TARGETING_TYPE, *COST_TYPE]
            df = pd.DataFrame(data=list(ability_vectors.values()), columns=columns, index=list(ability_vectors.keys()))
            df['form_count'] = list(ability_form_count.values())
        except:
            print(f"Error, couldn't complete: {champion_name}")
        else:
            print(f"Finished: {champion_name}")

    return df


if __name__ == "__main__":
    df = generate_ability_dataframe()
    print(df)
    df.to_csv('ability_df.csv')
