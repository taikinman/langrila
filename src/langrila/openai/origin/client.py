import os

from openai import AsyncOpenAI, OpenAI
from openai._streaming import AsyncStream, Stream
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from typing_extensions import override

from ...base import BaseClient
from ...utils import create_parameters


class OpenAIClient(BaseClient):
    def __init__(self, **kwargs):
        self._client = OpenAI(**create_parameters(OpenAI, **kwargs))
        self._async_client = AsyncOpenAI(**create_parameters(AsyncOpenAI, **kwargs))

    @override
    def generate_message(self, **kwargs) -> ChatCompletion | Stream[ChatCompletionChunk]:
        if not isinstance(kwargs.get("response_format", NOT_GIVEN), (NotGiven, dict)):
            completion_params = create_parameters(
                self._client.beta.chat.completions.parse, **kwargs
            )
            return self._client.beta.chat.completions.parse(**completion_params)
        else:
            completion_params = create_parameters(self._client.chat.completions.create, **kwargs)
            return self._client.chat.completions.create(**completion_params)

    @override
    async def generate_message_async(
        self, **kwargs
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        if not isinstance(kwargs.get("response_format", NOT_GIVEN), (NotGiven, dict)):
            completion_params = create_parameters(
                self._client.beta.chat.completions.parse, **kwargs
            )
            return await self._async_client.beta.chat.completions.parse(**completion_params)
        else:
            completion_params = create_parameters(self._client.chat.completions.create, **kwargs)
            return await self._async_client.chat.completions.create(**completion_params)

    def embed_text(self, **kwargs):
        return self._client.embeddings.create(**kwargs)

    def embed_text_async(self, **kwargs):
        return self._async_client.embeddings.create(**kwargs)
