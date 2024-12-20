from openai import OpenAI
import os
import json

CLASSIFIER_PROMPT = """
You are an advanced assistant specialized in analyzing and classifying descriptions of abilities from the game League of Legends. 
Your task is to read a textual description of a champion's ability and output a detailed JSON object containing only the properties that are true. 
Exclude any properties that are false or irrelevant.

### Predefined Properties to Extract
- is_aoe: The ability affects multiple targets in a specific area (e.g., "deals damage to enemies in a radius").
- is_skillshot: The ability requires aiming and can miss.
- is_multitarget: The ability requires aiming and can miss.
- is_hard_cc: The ability applies hard crowd control (e.g., stun, snare, knock-up, suppression).
- is_soft_cc: The ability applies soft crowd control (e.g., slow, blind, disarm, grounding).

### Additional Properties to Extract
Analyze the description for other properties that might be relevant to gameplay or mechanics. Here are examples:

- has_shield: The ability provides a shield to the caster or allies.
- heals: The ability heals the caster or allies.
- is_channel: The ability has a channeling period.
- is_global: The ability can be used anywhere on the map.
- has_movement: The ability provides mobility (e.g., dash, teleport, blink).
- has_damage_over_time: The ability applies damage over time (e.g., burn, poison).
- is_executable: The ability executes targets below a specific health threshold.
- scales_with_<RESOURCE>: The ability scales with <RESOURCE>
- deals_<TYPE>_dmg: The ability deals <TYPE> damage

### Output Format
Answer with JSON format and nothing else. 
Use the specific format:
{
    "property_name_1": true,
    "property_name_2": true,
    ...
}

### Example
Input: "This ability deals magic damage in a cone, slowing all enemies hit. It scales with the caster's ability power and provides a small shield to the caster when hitting at least one enemy."
Output:
{
    "is_aoe": true,
    "is_multitarget": true,
    "is_soft_cc": true,
    "has_shield": true,
    "scales_with_ap": true,
    "deals_magic_dmg": true
}

Input: "The champion dashes forward, dealing physical damage to all enemies in their path. If the ability hits an enemy champion, they are knocked up for 1 second."
{
    "is_aoe": true,
    "is_multitarget": true,
    "is_hard_cc": true,
    "has_movement": true,
    "scales_with_ad": true,
    "deals_physical_dmg": true
}
"""


def classify_ability(ability: str, context: str = "") -> dict:
    ability_prompt = f"""
    Context: {context}
    Abilty: {ability}
    """

    prompts = [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": ability_prompt},
    ]

    model = os.getenv("CLASSIFIER_MODEL")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=prompts,
        response_format={"type": "json_object"},
    )

    ability_str = response.choices[0].message.content

    try:
        ability_dict: dict = json.loads(ability_str)
    except json.JSONDecodeError:
        print(ability_str)
        raise

    return ability_dict
