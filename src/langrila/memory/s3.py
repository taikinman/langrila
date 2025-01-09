import hashlib
import json
import logging
import os
import secrets
from typing import Any, cast

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from ..core.memory import BaseConversationMemory


class S3ConversationMemory(BaseConversationMemory):
    """
    A conversation memory that stores the conversation history in an S3 bucket.
    To use this memory, you need to create an S3 bucket in advance.

    Parameters
    ----------
    bucket : str
        The name of the S3 bucket to store the conversation history.
    object_key : str, optional
        The key of the object to store the conversation history. If not provided, a random key is generated.
    region_name : str, optional
        The region name of the S3 bucket, by default None.
    api_version : str, optional
        The API version of the S3 client, by default None.
    use_ssl : bool, optional
        Whether to use SSL for the S3 client, by default True.
    verify : str or bool, optional
        Whether to verify the SSL certificate for the S3 client, by default None.
    endpoint_url_env_name : str, optional
        The environment variable name for the S3 endpoint URL, by default None.
    aws_access_key_id_env_name : str, optional
        The environment variable name for the AWS access key ID, by default None.
    aws_secret_access_key_env_name : str, optional
        The environment variable name for the AWS secret access key, by default None.
    aws_session_token_env_name : str, optional
        The environment variable name for the AWS session token, by default None.
    boto_config : BotoConfig, optional
        The configuration for the S3 client, by default None.
    """

    def __init__(
        self,
        bucket: str,
        *,
        object_key: str | None = None,
        region_name: str | None = None,
        api_version: str | None = None,
        use_ssl: bool = True,
        verify: str | bool | None = None,
        endpoint_url_env_name: str | None = None,
        aws_access_key_id_env_name: str | None = None,
        aws_secret_access_key_env_name: str | None = None,
        aws_session_token_env_name: str | None = None,
        boto_config: BotoConfig | None = None,
    ) -> None:
        self.bucket = bucket
        self.object_key = object_key
        self.region_name = region_name
        self.api_version = api_version
        self.use_ssl = use_ssl
        self.verify = verify
        self.endpoint_url = os.getenv(endpoint_url_env_name) if endpoint_url_env_name else None
        self.aws_access_key_id = (
            os.getenv(aws_access_key_id_env_name) if aws_access_key_id_env_name else None
        )
        self.aws_secret_access_key = (
            os.getenv(aws_secret_access_key_env_name) if aws_secret_access_key_env_name else None
        )
        self.aws_session_token = (
            os.getenv(aws_session_token_env_name) if aws_session_token_env_name else None
        )
        self.boto_config = boto_config

        self.s3_client = boto3.client(
            "s3",
            region_name=self.region_name,
            api_version=self.api_version,
            use_ssl=self.use_ssl,
            verify=self.verify,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            config=self.boto_config,
        )

        try:
            # check at minimum
            self.s3_client.head_bucket(Bucket=self.bucket)
            logging.info(f"s3 connect success. bucket_name: {self.bucket}")
        except Exception as e:
            logging.error(f"s3 connect failed:{e}")
            raise

    def store(self, conversation_history: list[list[dict[str, Any]]]) -> None:
        """
        Store the conversation history in an S3 bucket.

        Parameters
        ----------
        conversation_history : list[list[dict[str, Any]]]
            The conversation history to store. The outer list represents the conversation turns,
            and the inner list represents the messages in each turn.
        """
        if self.object_key is None:
            self.object_key = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
        try:
            json_data = json.dumps(conversation_history, ensure_ascii=False)
            self.s3_client.put_object(Bucket=self.bucket, Key=self.object_key, Body=json_data)
        except Exception as e:
            logging.error(f"s3 store failed: {e}")
            raise

    def load(self) -> list[list[dict[str, Any]]]:
        """
        Load the conversation history from an S3 bucket. If no history is found, return an empty list.
        The outer list represents the conversation turns, and the inner list represents the messages
        in each turn.

        Returns
        -------
        list[list[dict[str, Any]]]
            The conversation history. If no history is found, return an empty list.
        """
        if self.object_key is None:
            return []
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.object_key)
            content = response["Body"].read().decode("utf-8")
            conversation_history = json.loads(content)
            return cast(list[list[dict[str, Any]]], conversation_history)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                return []
            else:
                logging.error(f"s3 load failed: {e}")
                raise
        except Exception as e:
            logging.error(f"s3 load failedr: {e}")
            raise
