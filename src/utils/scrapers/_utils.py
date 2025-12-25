from dataclasses import dataclass
from markdownify import markdownify, MarkdownConverter
from typing import Dict, Optional
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from datetime import datetime
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
    leveling: str
    notes: Optional[str] = None
    blurb: Optional[str] = None
    tooltips: Tooltips = None
    cost: Optional[str] = None
    costtype: Optional[str] = None
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
    ontargetcd: Optional[str] = None
    recharge: Optional[str] = None
    affects: Optional[str] = None
    damagetype: Optional[str] = None
    spelleffects: Optional[str] = None
    spellshield: Optional[str] = None
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


# STS Betting Market Types Mapping
STS_MARKET_TYPES = {
    # Match Winner markets
    4070002: "match_winner",
    21030186: "match_winner_alt",
    18002001: "match_winner_special",
    5030244: "match_winner_handicap",
    10700000: "match_winner_main",
    5030241: "match_winner_spread",
    19370000: "match_winner_live",

    # Map Winner markets
    4070004: "map_winner",
    21030395: "map_winner_alt",
    18002007: "map_winner_special",
    10700008: "map_winner_main",

    # Map Winner Handicaps (range 13080100-13080140)
    13080100: "map_winner_handicap_0",
    13080110: "map_winner_handicap_10",
    13080120: "map_winner_handicap_20",
    13080130: "map_winner_handicap_30",
    13080140: "map_winner_handicap_40",

    # Map Count/Handicap markets
    5030242: "map_count_over_under",
    4070008: "map_handicap",
    21030328: "map_handicap_alt",
    10700004: "map_count_main",
    18002005: "map_count_special",
    19370060: "map_count_live",
}


@dataclass(slots=True, frozen=True)
class BettingOdds:
    """Individual betting outcome/odds for a specific result"""
    id_oppty: int
    oppty_type: int
    outcome: str
    odds: float
    is_active: bool
    outcome_id: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return dacite.from_dict(data_class=cls, data=data)


@dataclass(slots=True, frozen=True)
class BettingMarket:
    """Collection of odds for a specific market type"""
    market_type: str
    market_type_id: int
    odds_list: list[BettingOdds]
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return dacite.from_dict(data_class=cls, data=data)


@dataclass(slots=True, frozen=True)
class BettingMatch:
    """Complete match information with all betting markets"""
    match_id: int
    team1: str
    team2: str
    tournament: str
    start_time: datetime
    is_live: bool
    markets: dict[str, BettingMarket]
    match_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        config = dacite.Config(
            type_hooks={
                datetime: lambda x: datetime.fromisoformat(x) if isinstance(x, str) else x
            }
        )
        return dacite.from_dict(data_class=cls, data=data, config=config)
