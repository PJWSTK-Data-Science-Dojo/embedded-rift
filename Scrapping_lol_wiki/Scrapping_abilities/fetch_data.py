from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time


service = Service("C:\\Windows\\System32\\chromedriver.exe")
driver = webdriver.Chrome(service=service)


url = "https://wiki.leagueoflegends.com"

with open('Data\\table_data.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

urls = [url + lines.split(" - ")[1].strip() for lines in lines]
champions_stats = []
for url in urls:
    driver.get(url)
    time.sleep(8)
    champion_divs = driver.find_elements(By.CLASS_NAME, "lvlselect-champ")
    for champ_div in champion_divs:
        try:
            select_element = champ_div.find_element(By.TAG_NAME, "select")
            champion_name = select_element.get_dom_attribute("data-champ")
            select = Select(select_element)
            for option in select.options:
                try:
                    select.select_by_visible_text(option.text)
                    time.sleep(2)
                    stats_elements = driver.find_elements(By.CLASS_NAME, "infobox-section-two-cell")
                    stats_text = [stat.text for stat in stats_elements]
                    champions_stats.append({
                        "Champion": champion_name,
                        "Option": option.text,
                        "Stats": stats_text
                    })
                    print(f"Pobrano dane dla: {champion_name} ({option.text})")
                except NoSuchElementException:
                    continue
        except Exception:
            continue

import csv

with open("champions_stats_raw.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Champion", "Option", "Stats"])
    for champion in champions_stats:
        writer.writerow([champion["Champion"], champion["Option"], "; ".join(champion["Stats"])])
