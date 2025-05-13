# utils/data_fetcher.py
import json
import asyncio
from playwright.async_api import async_playwright

# Async Playwright JSON fetcher
async def fetch_json_data(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        pre_tag = await page.query_selector("pre")
        json_text = await pre_tag.text_content() if pre_tag else "{}"
        await browser.close()
        return json.loads(json_text)

# Sync wrapper
def fetch_json(url):
    return asyncio.run(fetch_json_data(url))

def fetch_season_json(tournament):
    seasons_url = f"https://api.sofascore.com/api/v1/tournament/{tournament['id']}/seasons"
    
    return fetch_json(seasons_url)

def fetch_standing_json(tournament, season):
    standings_url = (
            f"https://www.sofascore.com/api/v1/unique-tournament/"
            f"{tournament['unique_tournament']}/season/{season['id']}/standings/total"
        )
    
    return fetch_json(standings_url)