from openai import OpenAI
import os
import json
import numpy as np
from enum import StrEnum
import pandas as pd


def properties_dict_to_str(properties_dict: dict) -> str:
    return "\n".join([f"{key}: {value}" for key, value in properties_dict.items()])


def select_properties(selected_keys: list[str], properties_dict: dict) -> str:
    return "\n".join([f"{key}: {'true' if key in selected_keys else 'false'}" for key in properties_dict.keys()])


with open('../utils/ability2vec/properties_general.json') as f:
    properties_general_dict = json.load(f)

with open('../utils/ability2vec/properties_dmg_and_scaling.json') as f:
    properties_dmg_and_scaling_dict = json.load(f)

with open('../utils/ability2vec/properties_effects1.json') as f:
    properties_effects1_dict = json.load(f)

with open('../utils/ability2vec/properties_effects2.json') as f:
    properties_effects2_dict = json.load(f)

with open('../utils/ability2vec/properties_buffs.json') as f:
    properties_buffs_dict = json.load(f)

with open('../utils/ability2vec/properties_debuffs.json') as f:
    properties_debuffs_dict = json.load(f)

PROPERTIES_DICT = {**properties_general_dict, **properties_dmg_and_scaling_dict,
                   **properties_effects1_dict, **properties_effects2_dict,
                   **properties_buffs_dict, **properties_debuffs_dict}


class PROMPT(StrEnum):
    GENERAL = f"""
You are an advanced assistant specialized in analyzing and classifying abilities from the game League of Legends based on it's descriptions. 
Based on your extensive League of Legends knowledge and a list of predefined properties with it's descriptions your task 
is to output a JSON object containing information about which of the properties does the ability have.
Your specialty is classifying abilities based on general properties which provide basic information about the ability's behaviour and how it can be used.
You may only use predefined properties that are listed below and you must use their exact name.
If the property is irrelevant mark it as false.

### Predefined List of available Properties with Descriptions:
{properties_dict_to_str(properties_general_dict)}

### Output Format
Answer with JSON format and nothing else. 
Make sure you include ALL properties.
Your output doesn't need to be in alphabetical order, you can use the same order that you are given but you must consider every single property.
You can't add your own properties.
Use the specific format:
{{
    "property_name_1": true,
    "property_name_2": false,
    "property_name_3": true,
    ...
}}

### Example 1
<Input>
Decisive Strike
Active: Garen cleanses himself of all slows and gains 35% bonus movement speed for a duration.

Movement Speed Duration:
1 / 1.65 / 2.3 / 2.95 / 3.6
Additionally, Garen empowers his next basic attack within 4.5 seconds to have an uncancellable windup, lunge at the target, deal bonus physical damage, and silence them for 1.5 seconds.

Bonus Physical Damage:
30 / 60 / 90 / 120 / 150 (+ 50% AD)
Decisive Strike resets Garen's basic attack timer.
</Input>

<Output>
{select_properties(['is_active', 'is_self_cast'], properties_general_dict)}
</Output>

### Example 2
<Input>
Lucent Singularity
Active: Lux sends a lucent singularity to the target location, remaining there for 5 seconds to grant sight of the area and slow nearby enemies.

Slow:
25 / 30 / 35 / 40 / 45%
Lucent Singularity can be recast at any time while it is in flight or within the duration, and does so automatically after the duration.

Recast: Lux detonates the singularity, dealing magic damage to enemies within. If Lucent Singularity was recast while in flight, it will detonate upon arrival.

Magic Damage:
65 / 115 / 165 / 215 / 265 (+ 80% AP)
The slow lingers for 1 second after leaving the area. Enemies hit by Lucent Singularity's detonation are also slowed by the same amount for 1 second.
</Input>

<Output>
{select_properties(['is_aoe', 'is_multitarget', 'is_active', 'is_enemy_cast'], properties_general_dict)}
</Output>
    """
    DMG_AND_SCALING = f"""
You are an advanced assistant specialized in analyzing and classifying abilities from the game League of Legends based on it's descriptions. 
Based on your extensive League of Legends knowledge and a list of predefined properties with it's descriptions your task 
is to output a JSON object containing information about which of the properties does the ability have.
Your specialty is classifying abilities based on their damage type and scaling.
You may only use predefined properties that are listed below and you must use their exact name.
If the property is irrelevant mark it as false.

### Predefined List of available Properties with Descriptions:
{properties_dict_to_str(properties_dmg_and_scaling_dict)}

### Output Format
Answer with JSON format and nothing else. 
Make sure you include ALL properties.
Your output doesn't need to be in alphabetical order, you can use the same order that you are given but you must consider every single property.
You can't add your own properties.
Use the specific format:
{{
    "property_name_1": true,
    "property_name_2": false,
    "property_name_3": true,
    ...
}}

### Example 1
<Input>
Decisive Strike
Active: Garen cleanses himself of all slows and gains 35% bonus movement speed for a duration.

Movement Speed Duration:
1 / 1.65 / 2.3 / 2.95 / 3.6
Additionally, Garen empowers his next basic attack within 4.5 seconds to have an uncancellable windup, lunge at the target, deal bonus physical damage, and silence them for 1.5 seconds.

Bonus Physical Damage:
30 / 60 / 90 / 120 / 150 (+ 50% AD)
Decisive Strike resets Garen's basic attack timer.
</Input>

<Output>
{select_properties(['scaled_with_ad', 'deals_physical_damage'], properties_dmg_and_scaling_dict)}
</Output>

### Example 2
<Input>
Lucent Singularity
Active: Lux sends a lucent singularity to the target location, remaining there for 5 seconds to grant sight of the area and slow nearby enemies.

Slow:
25 / 30 / 35 / 40 / 45%
Lucent Singularity can be recast at any time while it is in flight or within the duration, and does so automatically after the duration.

Recast: Lux detonates the singularity, dealing magic damage to enemies within. If Lucent Singularity was recast while in flight, it will detonate upon arrival.

Magic Damage:
65 / 115 / 165 / 215 / 265 (+ 80% AP)
The slow lingers for 1 second after leaving the area. Enemies hit by Lucent Singularity's detonation are also slowed by the same amount for 1 second.
</Input>

<Output>
{select_properties(['scaled_with_ap', 'deals_magic_damage'], properties_dmg_and_scaling_dict)}
</Output>
    """
    EFFECTS1 = f"""
You are an advanced assistant specialized in analyzing and classifying abilities from the game League of Legends based on it's descriptions. 
Based on your extensive League of Legends knowledge and a list of predefined properties with it's descriptions your task 
is to output a JSON object containing information about which of the properties does the ability have.
Your specialty is classifying abilities based on their effects.
You may only use predefined properties that are listed below and you must use their exact name.
If the property is irrelevant mark it as false.

### Predefined List of available Properties with Descriptions:
{properties_dict_to_str(properties_effects1_dict)}

### Output Format
Answer with JSON format and nothing else. 
Make sure you include ALL properties.
Your output doesn't need to be in alphabetical order, you can use the same order that you are given but you must consider every single property.
You can't add your own properties.
Use the specific format:
{{
    "property_name_1": true,
    "property_name_2": false,
    "property_name_3": true,
    ...
}}

### Example 1
<Input>
Decisive Strike
Active: Garen cleanses himself of all slows and gains 35% bonus movement speed for a duration.

Movement Speed Duration:
1 / 1.65 / 2.3 / 2.95 / 3.6
Additionally, Garen empowers his next basic attack within 4.5 seconds to have an uncancellable windup, lunge at the target, deal bonus physical damage, and silence them for 1.5 seconds.

Bonus Physical Damage:
30 / 60 / 90 / 120 / 150 (+ 50% AD)
Decisive Strike resets Garen's basic attack timer.
</Input>

<Output>
{select_properties(['removes_slow'], properties_effects1_dict)}
</Output>

### Example 2
<Input>
Lucent Singularity
Active: Lux sends a lucent singularity to the target location, remaining there for 5 seconds to grant sight of the area and slow nearby enemies.

Slow:
25 / 30 / 35 / 40 / 45%
Lucent Singularity can be recast at any time while it is in flight or within the duration, and does so automatically after the duration.

Recast: Lux detonates the singularity, dealing magic damage to enemies within. If Lucent Singularity was recast while in flight, it will detonate upon arrival.

Magic Damage:
65 / 115 / 165 / 215 / 265 (+ 80% AP)
The slow lingers for 1 second after leaving the area. Enemies hit by Lucent Singularity's detonation are also slowed by the same amount for 1 second.
</Input>

<Output>
{select_properties(['has_soft_cc'], properties_effects1_dict)}
</Output>
    """
    EFFECTS2 = f"""
You are an advanced assistant specialized in analyzing and classifying abilities from the game League of Legends based on it's descriptions. 
Based on your extensive League of Legends knowledge and a list of predefined properties with it's descriptions your task 
is to output a JSON object containing information about which of the properties does the ability have.
Your specialty is classifying abilities based on their effects.
You may only use predefined properties that are listed below and you must use their exact name.
If the property is irrelevant mark it as false.

### Predefined List of available Properties with Descriptions:
{properties_dict_to_str(properties_effects2_dict)}

### Output Format
Answer with JSON format and nothing else. 
Make sure you include ALL properties.
Your output doesn't need to be in alphabetical order, you can use the same order that you are given but you must consider every single property.
You can't add your own properties.
Use the specific format:
{{
    "property_name_1": true,
    "property_name_2": false,
    "property_name_3": true,
    ...
}}

### Example 1
<Input>
Decisive Strike
Active: Garen cleanses himself of all slows and gains 35% bonus movement speed for a duration.

Movement Speed Duration:
1 / 1.65 / 2.3 / 2.95 / 3.6
Additionally, Garen empowers his next basic attack within 4.5 seconds to have an uncancellable windup, lunge at the target, deal bonus physical damage, and silence them for 1.5 seconds.

Bonus Physical Damage:
30 / 60 / 90 / 120 / 150 (+ 50% AD)
Decisive Strike resets Garen's basic attack timer.
</Input>

<Output>
{select_properties(['modifies_autoattacks', 'is_autoattack_reset'], properties_effects2_dict)}
</Output>

### Example 2
<Input>
Lucent Singularity
Active: Lux sends a lucent singularity to the target location, remaining there for 5 seconds to grant sight of the area and slow nearby enemies.

Slow:
25 / 30 / 35 / 40 / 45%
Lucent Singularity can be recast at any time while it is in flight or within the duration, and does so automatically after the duration.

Recast: Lux detonates the singularity, dealing magic damage to enemies within. If Lucent Singularity was recast while in flight, it will detonate upon arrival.

Magic Damage:
65 / 115 / 165 / 215 / 265 (+ 80% AP)
The slow lingers for 1 second after leaving the area. Enemies hit by Lucent Singularity's detonation are also slowed by the same amount for 1 second.
</Input>

<Output>
{select_properties(['has_persisting_effect'], properties_effects2_dict)}
</Output>
    """
    BUFFS = f"""
You are an advanced assistant specialized in analyzing and classifying abilities from the game League of Legends based on it's descriptions. 
Based on your extensive League of Legends knowledge and a list of predefined properties with it's description your task 
is to output a JSON object containing information about which of the properties does the ability have.
Your specialty is classifying abilities based on the buffs they provide. Consider both temporary and permanent ability effects.
You may only use predefined properties that are listed below and you must use their exact name.
If the property is irrelevant mark it as false.

### Predefined List of available Properties with Descriptions:
{properties_dict_to_str(properties_buffs_dict)}

### Output Format
Answer with JSON format and nothing else. 
Make sure you include ALL properties.
Your output doesn't need to be in alphabetical order, you can use the same order that you are given but you must consider every single property.
You can't add your own properties.
Use the specific format:
{{
    "property_name_1": true,
    "property_name_2": false,
    "property_name_3": true,
    ...
}}

### Example 1
<Input>
Bio-Arcane Barrage
Active: Kog'Maw empowers his basic attacks for the next 8 seconds to gain bonus attack range and deal bonus magic damage on-hit, capped at 100 against minions and monsters.
Bonus Attack Range:
130 / 150 / 170 / 190 / 210
Bonus Magic Damage:
3 / 3.75 / 4.5 / 5.25 / 6% (+ 1% per 100 AP) of target's maximum health
</Input>

<Output>
{select_properties(['buffs_attack_range'], properties_buffs_dict)}
</Output>

### Example 2
<Input>
Eye of the Storm
Passive: Whenever Janna's abilities slow or knock up at least one enemy champion, 20% of Eye of the Storm's cooldown is refunded. This may occur once per spell cast.

Active: Janna grants the target allied champion or turret, or herself, a shield for 4 seconds.
Shield Strength:
80 / 120 / 160 / 200 / 240 (+ 55% AP)
While the shield holds, the target gains  bonus attack damage.
Bonus Attack Damage:
10 / 15 / 20 / 25 / 30 (+ 10% AP)
</Input>

<Output>
{select_properties(['buffs_attack_damage'], properties_buffs_dict)}
</Output>
    """
    DEBUFFS = f"""
You are an advanced assistant specialized in analyzing and classifying abilities from the game League of Legends based on it's descriptions. 
Based on your extensive League of Legends knowledge and a list of predefined properties with it's description your task 
is to output a JSON object containing information about which of the properties does the ability have.
Your specialty is classifying abilities based on the debuffs they provide. Consider both temporary and permanent ability effects.
You may only use predefined properties that are listed below and you must use their exact name.
If the property is irrelevant mark it as false.

### Predefined List of available Properties with Descriptions:
{properties_dict_to_str(properties_debuffs_dict)}

### Output Format
Answer with JSON format and nothing else. 
Make sure you include ALL properties.
Your output doesn't need to be in alphabetical order, you can use the same order that you are given but you must consider every single property.
You can't add your own properties.
Use the specific format:
{{
    "property_name_1": true,
    "property_name_2": false,
    "property_name_3": true,
    ...
}}

### Example 1
<Input>
Caustic Spittle
Passive: Kog'Maw gains bonus attack speed.
Bonus Attack Speed:
10 / 15 / 20 / 25 / 30%

Active: Kog'Maw ejects a wad of spittle in the target direction that deals magic damage to the first enemy hit and reduces their armor and magic resistance for 4 seconds.
Magic Damage:
90 / 140 / 190 / 240 / 290 (+ 70% AP)
Resistances Reduction:
16 / 20 / 24 / 28 / 32%
</Input>

<Output>
{select_properties(['debuffs_armor', 'debuffs_magic_resists'], properties_debuffs_dict)}
</Output>

### Example 2
<Input>
Spirit Fire
Active: Nasus unleashes a spirit fire at the target location, granting sight of the area for 2.5 seconds and, after a 0.264 seconds delay, dealing magic damage to enemies within.
Magic Damage:
50 / 80 / 110 / 140 / 170 (+ 60% AP)
The fire then remains for 5 seconds, dealing magic damage each second to enemies within and inflicting them with armor reduction, lingering for 1 second.
Magic Damage Per Tick:»
10 / 16 / 22 / 28 / 34 (+ 12% AP)
Armor Reduction:
30 / 35 / 40 / 45 / 50% of target's armor
</Input>

<Output>
{select_properties(['debuffs_armor'], properties_debuffs_dict)}
</Output>
"""


def classify_ability(ability: str, prompt: PROMPT = PROMPT.GENERAL, context: str = "") -> dict:
    ability_prompt = f"""
    Context: {context}
    Abilty: {ability}
    """

    prompts = [
        {"role": "system", "content": prompt},
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


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
#
# def generate_weights(vectors: np.ndarray[np.ndarray], cutoff: int = 10, exponent: float = 2.0) -> np.ndarray:
#     # higher weight the more uncommon the property is
#     counts = vectors.sum(axis=0)
#     counts_norm = np.where(counts > cutoff, 1 - ((counts - cutoff) / (counts.max() - cutoff)), 1)
#     weights = np.exp2(counts_norm * exponent - exponent)
#     return weights

