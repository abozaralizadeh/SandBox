import logging
import subprocess
import typing

_playwright_installed = False


def _ensure_playwright():
    global _playwright_installed
    if _playwright_installed:
        return
    install = subprocess.run(["playwright", "install"])
    install_deps = subprocess.run(["playwright", "install-deps"])
    logging.info("Playwright install result: %s", install)
    logging.info("Playwright install-deps result: %s", install_deps)
    _playwright_installed = True


async def create_async_playwright_browser(
    headless: bool = True, args: typing.Optional[typing.List[str]] = None,
):
    _ensure_playwright()
    from playwright.async_api import async_playwright
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless, args=args)
    return browser, playwright


async def get_browse_web_tools():
    """Return (tools, aclose). Callers MUST ``await aclose()`` once finished so the
    Chromium subprocess and Playwright driver are torn down before the event loop
    closes — otherwise GC finalizes them later and raises 'Event loop is closed'."""
    from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
    async_browser, playwright = await create_async_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

    async def aclose():
        try:
            await async_browser.close()
        finally:
            await playwright.stop()

    return toolkit.get_tools(), aclose
