import hashlib
import os
import secrets
from typing import Any, cast

from azure.cosmos import CosmosClient, PartitionKey, exceptions

from ..core.memory import BaseConversationMemory


class CosmosConversationMemory(BaseConversationMemory):
    """
    A conversation memory that stores the conversation history in Azure Cosmos DB.
    To use this memory, you need to create a Cosmos DB in advance.
    When storing the conversation history, a container is created if it does not exist
    and the history is stored as an item in the container. If the item already exists,
    it is updated with the new history.

    Parameters
    ----------
    endpoint_env_name : str
        The environment variable name for the Cosmos DB endpoint.
    key_env_name : str
        The environment variable name for the Cosmos DB key.
    db_env_name : str
        The environment variable name for the Cosmos DB database name.
    container_name : str, optional
        The name of the container to store the conversation history.
        If not provided, a random name is generated, by default None.
    item_name : str, optional
        The name of the item to store the conversation history.
        If not provided, a random name is generated, by default None.
    partition_key : str, optional
        The partition key for the container, by default None.
    """

    def __init__(
        self,
        endpoint_env_name: str,
        key_env_name: str,
        db_env_name: str,
        container_name: str | None = None,
        item_name: str | None = None,
        partition_key: str | None = None,
    ) -> None:
        self.endpoint = os.getenv(endpoint_env_name)
        self.key = os.getenv(key_env_name)
        self.dbname = os.getenv(db_env_name)
        self.containername = (
            container_name
            if container_name
            else hashlib.sha256(secrets.token_bytes(32)).hexdigest()
        )
        self.itemname = (
            item_name if item_name else hashlib.sha256(secrets.token_bytes(32)).hexdigest()
        )
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
        """
        Store the conversation history in Azure Cosmos DB.

        Parameters
        ----------
        conversation_history : list[list[dict[str, Any]]]
            The conversation history to store. The outer list represents the conversation turns,
            and the inner list represents the messages in each turn.
        """
        item: dict[str, Any] = {}
        item["id"] = self.itemname
        item[self.partition_key] = self.partition_key
        item["history"] = conversation_history
        self.container.upsert_item(item)

        self.__stored = True

    def load(self) -> list[list[dict[str, Any]]]:
        """
        Load the conversation history from Azure Cosmos DB. If no history is found, return an empty list.
        The outer list represents the conversation turns, and the inner list represents the messages
        in each turn.

        Returns
        -------
        list[list[dict[str, Any]]]
            The conversation history. If no history is found, return an empty list.
        """
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
