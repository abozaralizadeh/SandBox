import os
from langchain_core.tools import tool
from AIBlog.azurestorage import upsert_title
from utils import get_flat_date, strtobool

@tool
def set_title(title: str) -> str:
    """set the title of the blog post"""
    try:
        if not strtobool(os.environ.get("DEBUG", False)):
            flat_date_hour = get_flat_date() + "_00"
            if not strtobool(os.environ.get("DEBUG", False)) or strtobool(os.environ.get("DEBUG_SAVE", False)):
                upsert_title(rowkey=flat_date_hour, title=title)
        return "Title set successfully"
    except Exception as e:
        return f"Error setting title: {e}"
    