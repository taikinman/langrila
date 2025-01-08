import os
from typing import Any, cast

from azure.cosmos import CosmosClient, PartitionKey, exceptions

from ..core.memory import BaseConversationMemory


class CosmosConversationMemory(BaseConversationMemory):
    def __init__(
        self,
        endpoint_env_name: str,
        key_env_name: str,
        db_env_name: str,
        container_name: str,
        item_name: str,
        partition_key: str | None = None,
    ) -> None:
        self.endpoint = os.getenv(endpoint_env_name)
        self.key = os.getenv(key_env_name)
        self.dbname = os.getenv(db_env_name)
        self.containername = container_name
        self.itemname = item_name
        self.partition_key = partition_key if partition_key else f"{container_name}"

        if not self.endpoint or not self.key or not self.dbname:
            raise ValueError("Please provide the endpoint, key, and database name for Cosmos DB.")

        # Create a Cosmos client
        client = CosmosClient(url=self.endpoint, credential=self.key)

        # Get a database
        database = client.get_database_client(database=self.dbname)

        # Get or create a container
        database.create_container_if_not_exists(
            id=self.containername,
            partition_key=PartitionKey(path=f"/{self.partition_key}"),
        )
        self.container = database.get_container_client(self.containername)

        self.__stored = False

    def store(self, conversation_history: list[list[dict[str, Any]]]) -> None:
        item: dict[str, Any] = {}
        item["id"] = self.itemname
        item[self.partition_key] = self.partition_key
        item["history"] = conversation_history
        self.container.upsert_item(item)

        self.__stored = True

    def load(self) -> list[list[dict[str, Any]]]:
        try:
            history = self.container.read_item(
                item=self.itemname, partition_key=self.partition_key
            )["history"]
            return cast(list[list[dict[str, Any]]], history)
        except exceptions.CosmosResourceNotFoundError:
            if not self.__stored:
                return []

            raise
        except Exception as e:
            raise e
