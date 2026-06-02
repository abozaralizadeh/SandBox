import xmltodict
import requests
import json
from langchain_core.tools import tool
from pydantic import BaseModel

RSS_URLS = {
    "en": "https://feeds.bbci.co.uk/news/rss.xml",
    "fa": "https://feeds.bbci.co.uk/persian/rss.xml",
    "it": "https://www.ansa.it/sito/ansait_rss.xml",
}

def getRSS(url: str) -> dict:
    response = requests.get(url)
    return xmltodict.parse(response.content)

def saveRSS(filepath: str, data: dict) -> None:
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def create_news_feed_tool(rss_url):
    @tool
    def get_todays_news_feed() -> list[dict]:
        """get todays news as a list of dicts containing title and description"""
        data = getRSS(rss_url)
        result = []
        items = data.get('rss', {}).get('channel', {}).get('item', [])
        if isinstance(items, dict):
            items = [items]
        for item in items:
            result.append({
                "title": item.get("title", ""),
                "description": item.get("description", ""),
            })
        return result
    return get_todays_news_feed

get_todays_news_feed = create_news_feed_tool(RSS_URLS["en"])
