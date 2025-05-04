from langchain_core.tools import tool
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
import os
from openai import AzureOpenAI
import json

from AIBlog.azurestorage import save_photo_to_blob

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_DALLE"],
    api_key=os.environ["AZURE_OPENAI_API_KEY_DALLE"],
)

@tool
def get_image_by_text(text: str) -> str:
    """get the url of an image created as the input text explains, the input text should describe well the expected output"""
    result = client.images.generate(
        model=os.environ.get("AZURE_OPENAI_MODEL_DALLE", "dall-e-3"), # the name of your DALL-E 3 deployment
        prompt=text,
        n=1
    )

    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    blob_image_url = save_photo_to_blob(image_url)
    return blob_image_url
