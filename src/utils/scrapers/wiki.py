import requests
import parsel
from ._utils import md
import time

champion = "Aphelios"
template = "Template:Data_{}/{}"
lol_wiki = "https://wiki.leagueoflegends.com/en-us/{}"
wiki_api = "https://wiki.leagueoflegends.com/en-us/api.php"

# Constants
MEDIA_KEYS = [
    "icon",
    "icon2",
    "icon3",
    "icon4",
    "icon5",  # Image-related keys
    "video",
    "video2",
    "yvideo",
    "yvideo2",  # Video-related keys
]
IGNORED_KEYS = [
    "disp_name",
    "cdstart",
    "ontargetcdstatic",
    "customlabel",
    "custominfo",
    "customlabel2",
    "custominfo2",
    "callforhelp",
    "flavorsound",
    "flavorvideo",
    "flavortext",
]


def add_tooltips(selector: parsel.Selector, tips: dict[str, dict[str, str]]):
    abilities = selector.xpath(".//*[@data-ability]")
    for tooltip in abilities:
        champion = tooltip.attrib["data-champion"].strip()
        ability = tooltip.attrib["data-ability"].strip()

        if ability in tips["abilities"]:
            continue

        params = {
            "action": "parse",
            "format": "json",
            "disablelimitreport": "true",
            "prop": "text",
            "contentmodel": "wikitext",
            "maxage": "600",
            "smaxage": "600",
            "text": f"{{{{Tooltip/Ability|champion={champion}|ability={ability}|game=lol}}}}",
        }

        resp = requests.get(wiki_api, params=params)
        if resp.status_code != 200:
            print(f"Failed to fetch ability tooltip {champion} {ability}")
            continue

        tooltip_html = resp.json()["parse"]["text"]["*"]
        tooltip_text = (
            parsel.Selector(tooltip_html).css(".blue-tooltip > div").getall()[-1]
        )
        tips["abilities"][ability] = md(tooltip_text).strip()

    data_tips = selector.xpath(".//*[@data-tip]")
    for tip in data_tips:
        data_tip = tip.attrib["data-tip"].strip()
        text = "".join(tip.css("::text").getall()).strip()
        if data_tip in tips["data"] or not text:
            continue

        params = {
            "action": "parse",
            "format": "json",
            "disablelimitreport": "true",
            "prop": "text",
            "contentmodel": "wikitext",
            "maxage": "600",
            "smaxage": "600",
            "text": f"{{{{Tooltip/Glossary|tip={data_tip}|game=lol}}}}",
        }

        resp = requests.get(wiki_api, params=params)
        if resp.status_code != 200:
            print(f"Failed to fetch tooltip {data_tip}")
            continue

        tooltip_html = resp.json()["parse"]["text"]["*"]
        tooltip_text = (
            parsel.Selector(tooltip_html).css(".blue-tooltip > div").getall()[-1]
        )
        tips["data"][data_tip] = md(tooltip_text).strip()


def get_champion_abilities(champion: str) -> list[str]:
    champion = champion.replace(" ", "_").replace("'", "%27")
    resp = requests.get(lol_wiki.format(champion))
    if resp.status_code != 200:
        print("Failed to fetch page")
        return []

    sel = parsel.Selector(resp.text)
    abilities = sel.css(".skill > div .ability-info-stats__ability::text").getall()
    return [ability.strip() for ability in abilities]


def get_skill_selector(champion: str, skill: str) -> parsel.Selector:
    skill_name = skill.split(",")[0]
    url = (
        lol_wiki.format(template.format(champion, skill_name))
        .replace(" ", "_")
        .replace("'", "%27")
    )
    time.sleep(5)
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Failed to fetch skill page {url}")
        print(resp.status_code)
        print(resp.text)

        return None

    return parsel.Selector(resp.text)


def handle_text_with_tooltips(
    key: str,
    selector: parsel.Selector,
    skill_data: dict,
    tooltips: dict[str, dict[str, str]],
) -> None:
    if key not in skill_data:
        skill_data[key] = ""

    text = "".join(selector.css("::text").getall()).strip()
    if not text:
        return

    text = md(selector.get()).strip()
    add_tooltips(selector, tooltips)

    if key not in skill_data:
        skill_data[key] = text
    else:
        skill_data[key] += f"\n\n{text}"


def _extract_ability_data(selector: parsel.Selector) -> dict:
    rows = selector.css("table.article-table.grid > tbody > tr")
    skill_data = {}
    tooltips = {"abilities": {}, "champions": {}, "items": {}, "data": {}}

    for row in rows[1:]:
        key_cell, value_cell, *_ = row.css(":scope > td")
        key = key_cell.css("code::text").get().strip()

        if key in MEDIA_KEYS or key in IGNORED_KEYS:
            continue

        if key.startswith("blurb"):
            handle_text_with_tooltips("blurb", value_cell, skill_data, tooltips)
            continue

        if key.startswith("desc"):
            handle_text_with_tooltips("description", value_cell, skill_data, tooltips)
            continue

        if key.startswith("leveling"):
            leveling_text = "".join(value_cell.css("::text").getall()).strip()

            if not leveling_text:
                continue

            skill_data["leveling"] = (
                skill_data.get("leveling", "") + f"\n{leveling_text}"
            )
            continue

        skill_data[key] = "".join(value_cell.css("::text").getall()).strip()

    skill_data["tooltips"] = tooltips
    return skill_data


def get_ability_data(champion: str, ability: str) -> dict:
    champion = champion.replace(" ", "_").replace("'", "")
    ability = ability.replace(" ", "_").replace("'", "%27")
    selector = get_skill_selector(champion, ability)
    if not selector:
        return {}

    return _extract_ability_data(selector)
