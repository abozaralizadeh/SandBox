import asyncio
import logging
import subprocess
import typing

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from playwright.async_api import Browser, async_playwright

install = subprocess.run(["playwright", "install"])
install_deps = subprocess.run(["playwright", "install-deps"])
logging.info("Playwright install result: %s", install)
logging.info("Playwright install-deps result: %s", install_deps)


async def create_async_playwright_browser(
    headless: bool = True, args: typing.Optional[typing.List[str]] = None
) -> Browser:
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless, args=args)
    return browser


async def get_browse_web_tools():
    async_browser = await create_async_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()
    return tools
