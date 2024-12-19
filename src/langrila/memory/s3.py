import hashlib
import json
import logging
import os
import secrets
from typing import Optional, Union

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from ..core.memory import BaseConversationMemory


class S3ConversationMemory(BaseConversationMemory):
    def __init__(
        self,
        bucket: str,
        *,
        object_key: Optional[str] = None,
        region_name: Optional[str] = None,
        api_version: Optional[str] = None,
        use_ssl: Optional[bool] = True,
        verify: Union[str, bool, None] = None,
        endpoint_url_env_name: Optional[str] = None,
        aws_access_key_id_env_name: Optional[str] = None,
        aws_secret_access_key_env_name: Optional[str] = None,
        aws_session_token_env_name: Optional[str] = None,
        boto_config: Optional[BotoConfig] = None,
    ):
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

    def store(self, conversation_history: list[dict[str, str]]):
        if self.object_key is None:
            self.object_key = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
        try:
            json_data = json.dumps(conversation_history, ensure_ascii=False)
            self.s3_client.put_object(Bucket=self.bucket, Key=self.object_key, Body=json_data)
        except Exception as e:
            logging.error(f"s3 store failed: {e}")
            raise

    def load(self) -> list[dict[str, str]]:
        if self.object_key is None:
            return []
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.object_key)
            content = response["Body"].read().decode("utf-8")
            conversation_history = json.loads(content)
            return conversation_history
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
