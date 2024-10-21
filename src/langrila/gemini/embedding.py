from typing import Any, Literal, Sequence

from google.auth import credentials as auth_credentials

from ..base import BaseEmbeddingModule
from ..result import EmbeddingResults


def get_embedding_module(
    model_name: str | None = None,
    title: str | None = None,
    api_type: Literal["genai", "vertexai"] = "genai",
    dimensions: int | None = None,
    task_type: str | None = None,
    api_key_env_name: str | None = None,
    project_id_env_name: str | None = None,
    location_env_name: str | None = None,
    experiment: str | None = None,
    experiment_description: str | None = None,
    experiment_tensorboard: str | bool | None = None,
    staging_bucket: str | None = None,
    credentials: auth_credentials.Credentials | None = None,
    encryption_spec_key_name: str | None = None,
    network: str | None = None,
    service_account: str | None = None,
    endpoint_env_name: str | None = None,
    request_metadata: Sequence[tuple[str, str]] | None = None,
) -> Any:
    if api_type == "genai":
        from .genai.embedding import GeminiAIStudioEmbeddingModule

        return GeminiAIStudioEmbeddingModule(
            model_name=model_name,
            api_key_env_name=api_key_env_name,
            title=title,
            dimensions=dimensions,
            task_type=task_type,
        )
    elif api_type == "vertexai":
        from .vertexai.embedding import VertexAIEmbeddingModule

        return VertexAIEmbeddingModule(
            model_name=model_name,
            title=title,
            dimensions=dimensions,
            task_type=task_type,
            api_key_env_name=api_key_env_name,
            project_id_env_name=project_id_env_name,
            location_env_name=location_env_name,
            experiment=experiment,
            experiment_description=experiment_description,
            experiment_tensorboard=experiment_tensorboard,
            staging_bucket=staging_bucket,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            network=network,
            service_account=service_account,
            endpoint_env_name=endpoint_env_name,
            request_metadata=request_metadata,
        )


class GeminiEmbeddingModule(BaseEmbeddingModule):
    def __init__(
        self,
        model_name: str | None = None,
        api_key_env_name: str | None = None,
        api_type: Literal["genai", "vertexai"] = "genai",
        title: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = None,
        project_id_env_name: str | None = None,
        location_env_name: str | None = None,
        experiment: str | None = None,
        experiment_description: str | None = None,
        experiment_tensorboard: str | bool | None = None,
        staging_bucket: str | None = None,
        credentials: auth_credentials.Credentials | None = None,
        encryption_spec_key_name: str | None = None,
        network: str | None = None,
        service_account: str | None = None,
        endpoint_env_name: str | None = None,
        request_metadata: Sequence[tuple[str, str]] | None = None,
        **kwargs: Any,
    ):
        self.api_type = api_type
        self._embed = get_embedding_module(
            model_name=model_name,
            api_key_env_name=api_key_env_name,
            api_type=api_type,
            title=title,
            dimensions=dimensions,
            task_type=task_type,
            project_id_env_name=project_id_env_name,
            location_env_name=location_env_name,
            experiment=experiment,
            experiment_description=experiment_description,
            experiment_tensorboard=experiment_tensorboard,
            staging_bucket=staging_bucket,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            network=network,
            service_account=service_account,
            endpoint_env_name=endpoint_env_name,
            request_metadata=request_metadata,
        )

    def run(
        self,
        text: str | list[str],
        model_name: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = None,
        title: str | None = None,
    ) -> EmbeddingResults:
        return self._embed.run(
            text=text,
            model_name=model_name,
            dimensions=dimensions,
            task_type=task_type,
            title=title,
        )

    async def arun(
        self,
        text: str | list[str],
        model_name: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = None,
        title: str | None = None,
    ) -> EmbeddingResults:
        return await self._embed.arun(
            text=text,
            model_name=model_name,
            dimensions=dimensions,
            task_type=task_type,
            title=title,
        )
