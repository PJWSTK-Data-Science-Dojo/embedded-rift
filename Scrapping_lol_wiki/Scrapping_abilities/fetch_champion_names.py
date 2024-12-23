import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

service = Service("C:\\Windows\\System32\\chromedriver.exe")
driver = webdriver.Chrome(service=service)

url = 'https://wiki.leagueoflegends.com/en-us/List_of_champions'
driver.get(url)

time.sleep(5)

table = driver.find_element(By.CLASS_NAME, "article-table")
rows = table.find_elements(By.TAG_NAME, "tr")
table_data = []

for row in rows:
    cells = row.find_elements(By.TAG_NAME, "td")
    if cells:
        first_cell_text = cells[0].text
        link_element = row.find_element(By.TAG_NAME, "a") if row.find_elements(By.TAG_NAME, "a") else None
        href_value = link_element.get_dom_attribute("href") if link_element else "Brak linku"
        table_data.append(f"{first_cell_text} - {href_value}")

with open("Data\\table_data.txt", "w", encoding="utf-8") as file:
    for item in table_data:
        file.write(item + "\n")

with open("Data\\table_data.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
modified_lines = []

for i in range(0, len(lines), 2):
    name_line = lines[i].strip()
    link_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
    link_content = link_line.split(" - ", 1)[-1]
    modified_lines.append(f"{name_line} - {link_content}")

with open("Data\\table_data.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(modified_lines))
driver.quit()