from azure.cosmos import exceptions, CosmosClient, PartitionKey
import logging
import os
import hashlib

from ..base import BaseConversationMemory

class CosmosConversationMemory(BaseConversationMemory):
    def __init__(self, endpoint_env_name: str, key_env_name: str, db_env_name: str, container_env_name: str):
        self.endpoint = os.getenv(endpoint_env_name)
        self.key = os.getenv(key_env_name)
        self.dbname = os.getenv(db_env_name)
        self.containername = os.getenv(container_env_name)
        # Create a Cosmos client
        try:
            client = CosmosClient(url=self.endpoint, credential=self.key)
        except:
            logging.error('Could not connect to Cosmos DB')
            raise ConnectionError
        # Get a database
        try:
            database = client.get_database_client(database=self.dbname)
        except exceptions.CosmosResourceNotFoundError:
            logging.error(f'Could not find database: {self.dbname}')
        # Get a container
        try:
            self.container = database.get_container_client(self.containername)
        except exceptions.CosmosResourceExistsError:
            logging.error(f'Could not find container: {self.containername}')

    def store(self, conversation_history: list[dict[str, str]]):
        for item in conversation_history:
            item["id"] = hashlib.sha256(str(item).encode()).hexdigest()
            try:
                self.container.create_item(item)
            except exceptions.CosmosResourceExistsError:
                pass
    
    def load(self) -> list[dict[str, str]]:
        result = []
        try:
            items = self.container.read_all_items()
            for item in items:
                result.append({k: v for k, v in item.items() if k in {"role", "content"}})
            return result
        except:
            return result
