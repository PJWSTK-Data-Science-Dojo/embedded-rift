from utils.ability2vec._classifier import classify_ability
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

if __name__ == "__main__":
    ability = """
    Innate: Xin Zhao's basic attacks and An icon for Xin Zhao's ability Wind Becomes Lightning Wind Becomes Lightning strikes generate a stack of Determination, stacking up to 3 times. 
    The third stack consumes them all to deal 15 / 30 / 45 / 60% (based on level) AD bonus physical damage and  heal Xin Zhao for 3 / 3.5 / 4% (based on level) of his maximum health (+ 65% AP).
    """

    data = classify_ability(ability)
    pprint(data)
