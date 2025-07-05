# from langchain_community.tools import DuckDuckGoSearchResults
# from langchain_community.tools import DuckDuckGoSearchRun

# searchinternettool = DuckDuckGoSearchResults(output_format="json")
# askinternettool = DuckDuckGoSearchRun()

import os
from langchain_community.tools.tavily_search import TavilySearchResults

searchinternettool = TavilySearchResults(tavily_api_key=os.environ["TAVILY_API_KEY"])
