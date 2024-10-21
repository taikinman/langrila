from typing import Any

from google.generativeai.embedding import EmbeddingTaskTypeOptions
from google.generativeai.types import helper_types, model_types

from ...base import BaseEmbeddingModule
from ...result import EmbeddingResults
from ...usage import Usage
from ..gemini_utils import get_client


class GeminiEmbeddingModule(BaseEmbeddingModule):
    def __init__(
        self,
        model_name: model_types.BaseModelNameOptions | None = None,
        title: str | None = None,
        dimensions: int | None = None,
        task_type: EmbeddingTaskTypeOptions | None = None,
        request_options: helper_types.RequestOptionsType | None = None,
        api_key_env_name: str | None = None,
        **kwargs: Any,
    ):
        self.api_type = "genai"
        self.model_name = model_name
        self.title = title
        self.dimensions = dimensions
        self.task_type = task_type
        self.request_options = request_options

        self._client = get_client(api_key_env_name=api_key_env_name, api_type=self.api_type)

    def _get_embedding_kwargs(self, **kwargs):
        _kwargs = {}
        _kwargs["model"] = kwargs.get("model_name") or self.model_name
        _kwargs["title"] = kwargs.get("title") or self.title

        if kwargs.get("task_type") or self.task_type:
            _kwargs["task_type"] = kwargs.get("task_type") or self.task_type

        _kwargs["request_options"] = kwargs.get("request_options") or self.request_options

        if kwargs.get("dimensions") or self.dimensions:
            _kwargs["output_dimensionality"] = kwargs.get("dimensions") or self.dimensions

        return _kwargs

    def run(
        self,
        text: str | list[str],
        model_name: model_types.BaseModelNameOptions | None = None,
        dimensions: int | None = None,
        task_type: EmbeddingTaskTypeOptions | None = None,
        request_options: helper_types.RequestOptionsType | None = None,
        title: str | None = None,
    ) -> EmbeddingResults:
        if not isinstance(text, list):
            text = [text]

        embedding_kwargs = self._get_embedding_kwargs(
            model_name=model_name,
            dimensions=dimensions,
            task_type=task_type,
            request_options=request_options,
            title=title,
        )

        embeddings = self._client.embed_text(content=text, **embedding_kwargs)
        prompt_usage = Usage(
            model_name=embedding_kwargs.get("model"),
            # NOTE: Counting tokens is not supported for the embedding models
            # prompt_tokens=self._client.count_tokens(
            #     contents=text, model_name=embedding_kwargs.get("model")
            # ),
        )

        return EmbeddingResults(text=text, embeddings=embeddings["embedding"], usage=prompt_usage)

    async def arun(
        self,
        text: str | list[str],
        model_name: model_types.BaseModelNameOptions | None = None,
        dimensions: int | None = None,
        task_type: EmbeddingTaskTypeOptions | None = None,
        request_options: helper_types.RequestOptionsType | None = None,
        title: str | None = None,
    ) -> EmbeddingResults:
        if not isinstance(text, list):
            text = [text]

        embedding_kwargs = self._get_embedding_kwargs(
            model_name=model_name,
            dimensions=dimensions,
            task_type=task_type,
            request_options=request_options,
            title=title,
        )

        embeddings = await self._client.embed_text_async(content=text, **embedding_kwargs)
        prompt_usage = Usage(
            model_name=embedding_kwargs.get("model"),
            # NOTE: Counting tokens is not supported for the embedding models
            # prompt_tokens=self._client.count_tokens(
            #     contents=text, model_name=embedding_kwargs.get("model")
            # ),
        )

        return EmbeddingResults(text=text, embeddings=embeddings["embedding"], usage=prompt_usage)
