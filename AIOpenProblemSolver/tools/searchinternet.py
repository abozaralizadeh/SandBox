import os

from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults

ddg_search_results = DuckDuckGoSearchResults(output_format="json")
ddg_search = DuckDuckGoSearchRun()

tavily_search = TavilySearchResults(
    max_results=10,
    include_answer=True,
    include_raw_content=True,
    include_images=False,
    tavily_api_key=os.environ["TAVILY_API_KEY"],
)
