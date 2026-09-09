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
    closes — otherwise GC finalizes them later and raises 'Event loop is closed'.

    Browsing is OPTIONAL. Azure App Service's Python image stopped shipping Chromium's
    system libraries on 2026-09-08 (`BrowserType.launch: Host system is missing dependencies
    to run browsers` — libglib2.0-0, libnss3, …), and `playwright install-deps` cannot fix it
    from inside the running container. With the launch raising, AIOPS died before it had
    an agent at all: AIBlog 500'd on every request and AIOPS's daily iteration failed silently.
    A missing browser must cost the browse tools, not the whole run — the hosted `web_search`
    tool and Tavily/DDG still work — so a launch failure degrades to no browse tools.
    """
    from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

    async def _noop_aclose():
        return None

    try:
        async_browser, playwright = await create_async_playwright_browser()
    except Exception as exc:  # noqa: BLE001 - any launch failure degrades to no browsing
        logging.warning(
            "Playwright browser unavailable (%s: %s); continuing without browse tools.",
            type(exc).__name__, str(exc).strip().splitlines()[0] if str(exc).strip() else "",
        )
        return [], _noop_aclose

    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

    async def aclose():
        try:
            await async_browser.close()
        finally:
            await playwright.stop()

    return toolkit.get_tools(), aclose
