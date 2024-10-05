import os

from openai import AsyncOpenAI, OpenAI
from openai._streaming import AsyncStream, Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from typing_extensions import override

from ...base import BaseClient
from ...utils import create_parameters


class OpenAIChat(BaseClient):
    def __init__(self, **kwargs):
        self._client = OpenAI(**create_parameters(OpenAI, **kwargs))
        self._async_client = AsyncOpenAI(**create_parameters(AsyncOpenAI, **kwargs))

    @override
    def generate_content(self, **kwargs) -> ChatCompletion | Stream[ChatCompletionChunk]:
        if not isinstance(kwargs.get("response_format", {}), dict):
            return self._client.beta.chat.completions.parse(**kwargs)
        else:
            return self._client.chat.completions.create(**kwargs)

    @override
    async def generate_content_async(
        self, **kwargs
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        if not isinstance(kwargs.get("response_format", {}), dict):
            return self._async_client.beta.chat.completions.parse(**kwargs)
        else:
            return self._async_client.chat.completions.create(**kwargs)
