from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.data.tables import TableServiceClient, TableEntity
from azure.data.tables import UpdateMode
from utils import get_flat_date_hour, get_flat_date_full
from io import BytesIO
from dotenv import load_dotenv
import requests
import os
import uuid

load_dotenv()
# Configuration
connection_string = os.getenv('connection_string')
container_name = os.getenv('aiblog_blob_name')
table_name = os.getenv('aiblog_table_name') 

# Create a TableServiceClient
service_client = TableServiceClient.from_connection_string(conn_str=connection_string)
# Get a reference to the table client
table_client = service_client.get_table_client(table_name)
# Create a BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# Create the container if it does not exist
container_client = blob_service_client.get_container_client(container_name)

def save_photo_to_blob(photo_url):
    """
    Downloads a photo from the given URL and uploads it to Azure Blob Storage.

    :param photo_url: URL of the photo to download.
    :param connection_string: Azure Blob Storage connection string.
    :param container_name: Name of the Azure Blob Storage container.
    :param blob_name: Name for the blob to save the photo as.
    :return: URL of the saved blob.
    """
    # Download the image
    response = requests.get(photo_url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    try:
        container_client.create_container()
    except Exception as e:
        # Container already exists, or other error
        pass

    # Upload the image to Blob Storage
    blob_name = blob_name = f"{get_flat_date_full()}_{uuid.uuid4()}.png"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(BytesIO(response.content), overwrite=True)

    # Construct the blob URL
    blob_url = blob_client.url

    return blob_url

def insert_history(rowkey, html_content):
    # Define the entity (row) to insert
    entity = {
        "PartitionKey": "getimagetool",  # Logical grouping for entities
        "RowKey": rowkey,                # Unique identifier within the partition
        "html_content": html_content
    }
    # Insert the entity
    try:
        table_client.create_entity(entity=entity)
        print("Row inserted successfully!")
    except Exception as e:
        print(f"Error inserting row: {e}")

def upsert_history(rowkey, html_content):
    # Define the entity (row) to insert
    entity = {
        "PartitionKey": "getimagetool",  # Logical grouping for entities
        "RowKey": rowkey,                # Unique identifier within the partition
        "html_content": html_content
    }
    # Insert the entity
    try:
        table_client.upsert_entity(entity=entity, mode=UpdateMode.MERGE)
        print("Row upserted successfully!")
    except Exception as e:
        print(f"Error inserting row: {e}")

def insert_title(rowkey, title):
    # Define the entity (row) to insert
    entity = {
        "PartitionKey": "getimagetool",  # Logical grouping for entities
        "RowKey": rowkey,                # Unique identifier within the partition
        "title": title
    }
    # Insert the entity
    try:
        table_client.create_entity(entity=entity)
        print("Row inserted successfully!")
    except Exception as e:
        print(f"Error inserting row: {e}")

def upsert_title(rowkey, title):
    # Define the entity (row) to insert
    entity = {
        "PartitionKey": "getimagetool",  # Logical grouping for entities
        "RowKey": rowkey,                # Unique identifier within the partition
        "title": title
    }
    # Insert the entity
    try:
        table_client.upsert_entity(entity=entity, mode=UpdateMode.MERGE)
        print("Row upserted successfully!")
    except Exception as e:
        print(f"Error inserting row: {e}")


def get_last_n_rows(n=10):
    try:
        # Query all entities
        partition_key = "getimagetool"
        entities = table_client.query_entities(query_filter=f"PartitionKey eq '{partition_key}'", results_per_page=1000)

        # Convert the entities to a sorted list by RowKey (descending order)
        sorted_entities = sorted(
            entities,
            key=lambda x: x["RowKey"],  # Sort by RowKey
            reverse=False               # Ascending order
        )

        last_n_rows = [
            {key: value for key, value in row.items() if key not in ["PartitionKey", "RowKey"]}
            for row in sorted_entities[:n]
        ]

        return last_n_rows

    except Exception as e:
        print(f"Error retrieving rows: {e}")
        return None
    

def get_row(rowkey):
    try:
        # Retrieve the entity (row) using the PartitionKey and RowKey
        entity = table_client.get_entity(partition_key="getimagetool", row_key=rowkey)
        print(f"Row retrieved successfully: {entity}")
        return entity
    except Exception as e:
        print(f"Error retrieving row: {e}")
        return None