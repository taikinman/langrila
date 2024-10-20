import os
from typing import Mapping

import httpx
from openai.lib.azure import AzureADTokenProvider
from openai.resources.embeddings import Embeddings

from ..base import BaseEmbeddingModule
from ..result import EmbeddingResults
from ..usage import Usage
from ..utils import make_batch
from .openai_utils import get_client


class OpenAIEmbeddingModule(BaseEmbeddingModule):
    def __init__(
        self,
        api_key_env_name: str,
        organization_id_env_name: str | None = None,
        model_name: str | None = "text-embedding-3-small",
        dimensions: int | None = None,
        user: str | None = None,
        api_type: str | None = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        max_retries: int = 5,
        timeout: int = 60,
        batch_size: int = 10,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AzureADTokenProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
    ):
        assert api_type in ["openai", "azure"], "api_type must be 'openai' or 'azure'."
        if api_type == "azure":
            assert (
                api_version and endpoint_env_name and deployment_id_env_name
            ), "api_version, endpoint_env_name, and deployment_id_env_name must be specified for Azure API."

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
            self.additional_params["dimensions"] = dimensions

        if user is not None:
            self.additional_params["user"] = user

        self._client = get_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
            project=project,
            base_url=base_url,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )

    def run(self, text: str | list[str]) -> EmbeddingResults:
        if not isinstance(text, list):
            text = [text]

        embeddings = []
        total_usage = Usage(model_name=self.model_name)
        for batch in make_batch(text, batch_size=self.batch_size):
            response = self._client.embed_text(
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

    async def arun(self, text: str) -> EmbeddingResults:
        if not isinstance(text, list):
            text = [text]

        embeddings = []
        total_usage = Usage(model_name=self.model_name)
        for batch in make_batch(text, batch_size=self.batch_size):
            response = await self._client.embed_text_async(
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
