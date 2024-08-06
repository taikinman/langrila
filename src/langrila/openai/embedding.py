from openai.resources.embeddings import Embeddings

from ..base import BaseEmbeddingModule
from ..result import EmbeddingResults
from ..usage import Usage
from ..utils import make_batch
from .model_config import _NEWER_EMBEDDING_CONFIG, EMBEDDING_CONFIG
from .openai_utils import get_async_client, get_client


class OpenAIEmbeddingModule(BaseEmbeddingModule):
    def __init__(
        self,
        api_key_env_name: str,
        organization_id_env_name: str | None = None,
        model_name: str = "text-embedding-ada-002",
        dimensions: int | None = None,
        user: str | None = None,
        api_type: str | None = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        max_retries: int = 5,
        timeout: int = 60,
        batch_size: int = 10,
    ):
        assert api_type in ["openai", "azure"], "api_type must be 'openai' or 'azure'."
        if api_type == "azure":
            assert (
                api_version and endpoint_env_name and deployment_id_env_name
            ), "api_version, endpoint_env_name, and deployment_id_env_name must be specified for Azure API."

        assert (
            model_name in EMBEDDING_CONFIG.keys()
        ), f"model_name must be one of {', '.join(sorted(EMBEDDING_CONFIG.keys()))}."

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

        self.additional_params = {}
        if dimensions is not None:
            if model_name in _NEWER_EMBEDDING_CONFIG:
                self.additional_params["dimensions"] = dimensions
            else:
                print(f"Warning: dimensions is not supported for {model_name}. It will be ignored.")
        if user is not None:
            self.additional_params["user"] = user

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
        total_usage = Usage(model_name=self.model_name)
        for batch in make_batch(text, batch_size=self.batch_size):
            response = embedder.create(input=batch, model=self.model_name, **self.additional_params)
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
        total_usage = Usage(model_name=self.model_name)
        for batch in make_batch(text, batch_size=self.batch_size):
            response = await embedder.create(
                input=batch, model=self.model_name, **self.additional_params
            )
            embeddings.extend([e.embedding for e in response.data])
            total_usage += response.usage

        results = EmbeddingResults(
            text=text,
            embeddings=embeddings,
            usage=total_usage,
        )
        return results
