from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import DuckDuckGoSearchRun

ddgsearchinternettool = DuckDuckGoSearchResults(output_format="json")
ddgaskinternettool = DuckDuckGoSearchRun()

import os
from langchain_community.tools.tavily_search import TavilySearchResults

tavilysearchinternettool = TavilySearchResults(
                max_results=10,
                include_answer=True,
                include_raw_content=True,
                include_images=False,
                tavily_api_key=os.environ["TAVILY_API_KEY"])
