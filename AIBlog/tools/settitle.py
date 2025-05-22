import os
from langchain_core.tools import tool
from AIBlog.azurestorage import insert_title
from utils import get_flat_date, strtobool

@tool
def set_title(title: str) -> str:
    """set the title of the blog post"""
    try:
        if not strtobool(os.environ.get("DEBUG", False)):
            flat_date_hour = get_flat_date() + "_00"
            insert_title(rowkey=flat_date_hour, title=title)
        return "Title set successfully"
    except Exception as e:
        return f"Error setting title: {e}"
    