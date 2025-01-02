import json
import parsel

from utils.api_handler import APIHandler
from ._utils import md

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


class WikiScraper(APIHandler):
    def __init__(self):
        super().__init__(rate_window=30, rate_limit=30)
        self._cache = {}

    def get_champion_ability_names(self, champion: str) -> list[str]:
        champion = champion.replace(" ", "_").replace("'", "%27")
        response = self.send_request(lol_wiki.format(champion))

        sel = parsel.Selector(response.text)
        abilities = sel.css(".skill > div .ability-info-stats__ability::text").getall()
        return [ability.strip() for ability in abilities]

    def get_ability_data(self, champion: str, ability: str) -> dict:
        champion = champion.replace(" ", "_").replace("'", "%27")
        ability = ability.replace(" ", "_").replace("'", "%27")
        selector = self._get_skill_selector(champion, ability)
        return self._extract_ability_data(selector)

    def _extract_ability_data(self, selector: parsel.Selector) -> dict:
        rows = selector.css("table.article-table.grid > tbody > tr")
        skill_data = {}
        tooltips = {"abilities": {}, "champions": {}, "items": {}, "data": {}}

        for row in rows[1:]:
            key_cell, value_cell, *_ = row.css(":scope > td")
            key = key_cell.css("code::text").get().strip()

            if key in MEDIA_KEYS or key in IGNORED_KEYS:
                continue

            if key.startswith("blurb"):
                self._handle_text_with_tooltips(
                    "blurb", value_cell, skill_data, tooltips
                )
                continue

            if key.startswith("desc"):
                self._handle_text_with_tooltips(
                    "description", value_cell, skill_data, tooltips
                )
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

    def _handle_ability_tooltip(
        self, tooltip: parsel.Selector, tips: dict[str, dict[str, str]]
    ):
        champion = tooltip.attrib["data-champion"].strip()
        ability = tooltip.attrib["data-ability"].strip()

        if ability in tips["abilities"]:
            return

        if ability in self._cache:
            text = self._cache[ability]
            tips["abilities"][ability] = text
            return

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

        resp = self.send_request(wiki_api, params=params)
        if resp.status_code != 200:
            print(f"Failed to fetch ability tooltip {champion} {ability}")
            return

        tooltip_html = resp.json()["parse"]["text"]["*"]
        tooltip_text = (
            parsel.Selector(tooltip_html).css(".blue-tooltip > div").getall()[-1]
        )
        text = md(tooltip_text).strip()

        self._cache[ability] = text
        tips["abilities"][ability] = text

    def _handle_data_tooltip(
        self, tip: parsel.Selector, tips: dict[str, dict[str, str]]
    ):
        data_tip = tip.attrib["data-tip"].strip()
        text = "".join(tip.css("::text").getall()).strip()
        if data_tip in tips["data"] or not text:
            return

        if data_tip in self._cache:
            text = self._cache[data_tip]
            tips["data"][data_tip] = text
            return

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

        resp = self.send_request(wiki_api, params=params)
        if resp.status_code != 200:
            print(f"Failed to fetch tooltip {data_tip}")
            return

        tooltip_html = resp.json()["parse"]["text"]["*"]
        tooltip_text = (
            parsel.Selector(tooltip_html).css(".blue-tooltip > div").getall()[-1]
        )
        text = md(tooltip_text).strip()
        self._cache[data_tip] = text
        tips["data"][data_tip] = text

    def _add_tooltips(self, selector: parsel.Selector, tips: dict[str, dict[str, str]]):
        abilities = selector.xpath(".//*[@data-ability]")
        for tooltip in abilities:
            self._handle_ability_tooltip(tooltip, tips)

        data_tips = selector.xpath(".//*[@data-tip]")
        for tip in data_tips:
            self._handle_data_tooltip(tip, tips)

        # TODO: Add item and champion tooltips

    def _get_skill_selector(self, champion: str, skill: str) -> parsel.Selector:
        skill_name = skill.split(",")[0]
        url = (
            lol_wiki.format(template.format(champion, skill_name))
            .replace(" ", "_")
            .replace("'", "%27")
        )

        resp = self.send_request(url)
        if resp.status_code != 200:
            print(f"Failed to fetch skill page {url}")
            print(resp.status_code)
            print(resp.text)
            raise Exception("Failed to fetch skill page")

        return parsel.Selector(resp.text)

    def _handle_text_with_tooltips(
        self,
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
        self._add_tooltips(selector, tooltips)

        if key not in skill_data:
            skill_data[key] = text
        else:
            skill_data[key] += f"\n\n{text}"

    def save_cache(self) -> None:
        with open("data/cache.json", "w") as f:
            json.dump(self._cache, f, indent=4)
