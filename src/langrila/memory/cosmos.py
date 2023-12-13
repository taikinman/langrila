from azure.cosmos import exceptions, CosmosClient, PartitionKey
import logging
import os
import uuid

from ..base import BaseConversationMemory

class CosmosConversationMemory(BaseConversationMemory):
    def __init__(self, endpoint_name: str, key_name: str, dbname_name: str, containername_name: str, pkey_name: str):
        self.endpoint = os.getenv(endpoint_name)
        self.key = os.getenv(key_name)
        self.dbname = os.getenv(dbname_name)
        self.containername = os.getenv(containername_name)
        self.pkey = '/' + os.getenv(pkey_name).strip('/')
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
            item['id'] = str(uuid.uuid4())
            self.container.create_item(item)

    def load(self) -> list[dict[str, str]]:
        result = []
        try:
            items = self.container.read_items()
            for item in items:
                result.append(item)
            return result
        except:
            return result
