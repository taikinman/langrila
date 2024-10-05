from openai import AsyncAzureOpenAI, AzureOpenAI
from openai._streaming import AsyncStream, Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from typing_extensions import override

from ...base import BaseClient
from ...utils import create_parameters


class AzureOpenAIChat(BaseClient):
    def __init__(
        self,
        **kwargs,
    ):
        self._client = AzureOpenAI(**create_parameters(AzureOpenAI, **kwargs))
        self._async_client = AsyncAzureOpenAI(**create_parameters(AsyncAzureOpenAI, **kwargs))

    @override
    def generate_content(
        self,
        **kwargs,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        if not isinstance(kwargs.get("response_format", {}), dict):
            completion_params = create_parameters(
                self._client.beta.chat.completions.parse, **kwargs
            )
            return self._client.beta.chat.completions.parse(**completion_params)
        else:
            completion_params = create_parameters(self._client.chat.completions.create, **kwargs)
            return self._client.chat.completions.create(**completion_params)

    @override
    async def generate_content_async(
        self,
        **kwargs,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        if not isinstance(kwargs.get("response_format", {}), dict):
            completion_params = create_parameters(
                self._async_client.beta.chat.completions.parse, **kwargs
            )
            return await self._client.beta.chat.completions.parse(**completion_params)
        else:
            completion_params = create_parameters(
                self._async_client.chat.completions.create, **kwargs
            )
            return await self._async_client.chat.completions.create(**completion_params)
