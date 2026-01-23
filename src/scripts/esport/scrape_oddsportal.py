import asyncio
import json
import base64
import hashlib
import tqdm
from typing import List, Dict, Any
from urllib.parse import urljoin
from playwright.async_api import async_playwright, Browser, Page
from Crypto.Cipher import AES
from collections import deque

BASE_URL = "https://www.oddsportal.com"
TOURNAMENTS_URL = "https://www.oddsportal.com/pl/esports/results/"
PASSWORD = "J*8sQ!p$7aD_fR2yW@gHn*3bVp#sAdLd_k"
SALT = "5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d"
OUTPUT_FILE = "matches.json"

GOOD_LEAGUES: List[str] = [
    "lcs",
    "lec",
    "first-stand",
    "msi",
    "lck",
    "lpl",
    "lta",
    "mistrzostwa-swiata",
    "mid-season-invitational",
]


def decrypt_server_response(enc: str, password: str, salt: str) -> str:
    try:
        raw = base64.b64decode(enc)
        decoded = raw.decode("utf-8")
        cipher_b64, iv_hex = decoded.split(":", 1)
        iv = bytes.fromhex(iv_hex)
        cipher_bytes = base64.b64decode(cipher_b64)

        key = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 1000, dklen=32
        )

        cipher = AES.new(key, AES.MODE_CBC, iv)
        plaintext_padded = cipher.decrypt(cipher_bytes)

        pad_len = plaintext_padded[-1]
        if pad_len < 1 or pad_len > 16:
            raise ValueError("Invalid padding")

        plaintext = plaintext_padded[:-pad_len]
        return plaintext.decode("utf-8")
    except Exception:
        return "{}"


async def get_tournament_links(browser: Browser) -> List[str]:
    page = await browser.new_page()
    await page.goto(TOURNAMENTS_URL)

    await page.wait_for_selector('a[href*="league-of-legends"]', timeout=10000)

    links: List[str] = await page.eval_on_selector_all(
        'a[href*="league-of-legends"]',
        'elements => elements.map(e => e.getAttribute("href"))',
    )
    await page.close()

    clean_links: List[str] = []
    for link in links:
        link = link.replace("//", "/league-of-legends/")
        full_link = urljoin(BASE_URL, link)

        if any(gl in full_link for gl in GOOD_LEAGUES):
            clean_links.append(full_link)
        # clean_links.append(urljoin(BASE_URL, link))

    return list(set(clean_links))


async def fetch_rows_from_response(
    page: Page, action_coroutine: Any
) -> List[Dict[str, Any]]:
    async with page.expect_response(
        lambda r: "archive" in r.url and r.status == 200, timeout=10000
    ) as response_info:
        await action_coroutine
        response = await response_info.value
        body = await response.body()

        decrypted = decrypt_server_response(body.decode(), PASSWORD, SALT)
        data = json.loads(decrypted)

        return data.get("d", {}).get("rows", [])


async def process_tournament(
    browser: Browser, tournament_url: str
) -> List[Dict[str, Any]]:
    page = await browser.new_page()
    matches: List[Dict[str, Any]] = []
    await page.goto(tournament_url)

    print(f"Processing: {tournament_url}")
    all_years = await page.query_selector_all(
        "a.flex.items-center.justify-center.h-8.px-3.bg-gray-medium.cursor-pointer"
    )
    hrefs = [await t.get_attribute("href") for t in all_years]
    links = [urljoin(BASE_URL, h) for h in hrefs if h]
    print(f"  -> Found {len(links)} year(s) to process")
    try:
        for link in links:
            page_num = 1

            print(f"Fetching tournament {link}")
            print(f"  -> Page {page_num}: fetching...")
            rows = await fetch_rows_from_response(page, page.goto(link))
            matches.extend(rows)
            await page.wait_for_timeout(2000)

            print(f"  -> Page {page_num}: found {len(rows)} matches")
            buttons = await page.query_selector_all("a.pagination-link")
            if not buttons:
                continue

            _, *page_btns, next = buttons

            queue = deque(page_btns)
            print(f"  -> Total pages: {len(page_btns) + 1}")
            while queue:
                next_btn = queue.popleft()
                text = await next_btn.inner_text()
                print(f"  -> Page {text}: fetching...")
                try:
                    rows = await fetch_rows_from_response(page, next_btn.click())
                    matches.extend(rows)
                    print(f"  -> Page {text}: found {len(rows)} matches")

                    await page.wait_for_timeout(1000)
                except Exception as e:
                    queue.append(next_btn)
                    await page.wait_for_timeout(2000)

    except Exception as e:
        print(f"Error processing tournament {tournament_url}: {e}")
        raise e
    finally:
        await page.close()

    return matches


async def main() -> None:
    all_matches: List[Dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        print("Fetching tournament list...")
        tournament_links = await get_tournament_links(browser)
        print(f"Found {len(tournament_links)} tournaments to scrape.")

        pbar = tqdm.tqdm(tournament_links)

        for link in pbar:
            pbar.set_description(f"Scraping {link.split('/')[-2]}")

            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    tournament_matches = await process_tournament(browser, link)
                    all_matches.extend(tournament_matches)
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"\nError (Attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        await asyncio.sleep(10)

    print(f"\nTotal matches fetched: {len(all_matches)}")

    unique_matches = {m["id"]: m for m in all_matches}.values()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(list(unique_matches), f, indent=4)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
