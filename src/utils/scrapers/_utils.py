from dataclasses import dataclass
from markdownify import markdownify, MarkdownConverter
from dataclasses import dataclass
from typing import Dict, Optional, Self
import dacite


@dataclass(slots=True, frozen=True)
class Tooltips:
    abilities: dict[str, str]
    champions: dict[str, str]
    items: dict[str, str]
    data: dict[str, str]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return dacite.from_dict(data_class=cls, data=data)


@dataclass(slots=True, frozen=True)
class ChampionAbility:
    description: str
    notes: Optional[str] = None
    blurb: Optional[str] = None
    tooltips: Tooltips = None
    cost: Optional[str] = None
    cost_type: Optional[str] = None
    targeting: Optional[str] = None
    cooldown: Optional[str] = None
    skill: Optional[str] = None
    range: Optional[str] = None
    target_range: Optional[str] = None
    attack_range: Optional[str] = None
    collision_radius: Optional[str] = None
    effect_radius: Optional[str] = None
    width: Optional[str] = None
    angle: Optional[str] = None
    inner_radius: Optional[str] = None
    tether_radius: Optional[str] = None
    speed: Optional[str] = None
    cast_time: Optional[str] = None
    static: Optional[str] = None
    on_target_cd: Optional[str] = None
    recharge: Optional[str] = None
    affects: Optional[str] = None
    damage_type: Optional[str] = None
    spell_effects: Optional[str] = None
    spell_shield: Optional[str] = None
    projectile: Optional[str] = None
    grounded: Optional[str] = None
    knockdown: Optional[str] = None
    silence: Optional[str] = None
    additional: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return dacite.from_dict(data_class=cls, data=data)


@dataclass(slots=True, frozen=True)
class ChampionStats:
    hp: float
    hpperlevel: float
    mp: float
    mpperlevel: float
    movespeed: float
    armor: float
    armorperlevel: float
    spellblock: float
    spellblockperlevel: float
    attackrange: float
    hpregen: float
    hpregenperlevel: float
    mpregen: float
    mpregenperlevel: float
    crit: float
    critperlevel: float
    attackdamage: float
    attackdamageperlevel: float
    attackspeedperlevel: float
    attackspeed: float

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return dacite.from_dict(data_class=cls, data=data)


@dataclass(slots=True, frozen=True)
class Champion:
    name: str
    stats: ChampionStats
    abilities: dict[str, ChampionAbility]
    resource: str
    patch: str

    @classmethod
    def from_dict(cls, data: dict) -> "Champion":
        return dacite.from_dict(data_class=cls, data=data)


class _CustomMarkdownConverter(MarkdownConverter):
    def convert_a(self, el, text, convert_as_inline):
        # Ignore <a> tags; just return the text content
        return text

    def convert_img(self, el, text, convert_as_inline):
        # Ignore <img> tags; return an empty string or some placeholder if needed
        return ""


def md(html):
    return _CustomMarkdownConverter().convert(html)
