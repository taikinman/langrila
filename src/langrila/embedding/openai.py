from typing import Optional

from openai.resources.embeddings import Embeddings

from ..base import BaseModule
from ..result import EmbeddingResults
from ..usage import Usage
from ..utils import get_async_client, get_client, make_batch


class OpenAIEmbeddingModule(BaseModule):
    def __init__(
        self,
        api_key_env_name: str,
        organization_id_env_name: Optional[str] = None,
        model_name: str = "text-embedding-ada-002",
        api_type: Optional[str] = "openai",
        api_version: Optional[str] = None,
        endpoint_env_name: Optional[str] = None,
        deployment_id_env_name: Optional[str] = None,
        max_retries: int = 5,
        timeout: int = 60,
        batch_size: int = 10,
    ):
        self.api_key_env_name = api_key_env_name
        self.organization_id_env_name = organization_id_env_name
        self.model_name = model_name
        self.api_type = api_type
        self.api_version = api_version
        self.endpoint_env_name = endpoint_env_name
        self.deployment_id_env_name = deployment_id_env_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = batch_size

    def run(self, text: str | list[str]) -> EmbeddingResults:
        client = get_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        embedder = Embeddings(client)

        if not isinstance(text, list):
            text = [text]

        embeddings = []
        total_usage = Usage()
        for batch in make_batch(text, batch_size=self.batch_size):
            response = embedder.create(input=batch, model=self.model_name)
            embeddings.extend([e.embedding for e in response.data])
            total_usage += response.usage

        results = EmbeddingResults(
            text=text,
            embeddings=embeddings,
            usage=total_usage,
        )
        return results

    async def arun(self, text: str) -> EmbeddingResults:
        client = get_async_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        embedder = Embeddings(client)

        if not isinstance(text, list):
            text = [text]

        embeddings = []
        total_usage = Usage()
        for batch in make_batch(text, batch_size=self.batch_size):
            response = await embedder.create(input=batch, model=self.model_name)
            embeddings.extend([e.embedding for e in response.data])
            total_usage += response.usage

        results = EmbeddingResults(
            text=text,
            embeddings=embeddings,
            usage=total_usage,
        )
        return results
