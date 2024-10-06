from anthropic import Anthropic, AsyncAnthropic
from anthropic._streaming import AsyncStream, Stream
from anthropic.types import Message, RawMessageStreamEvent
from typing_extensions import override

from ...base import BaseClient
from ...utils import create_parameters


class ClaudeAnthropicClient(BaseClient):
    def __init__(self, **kwargs):
        self._client = Anthropic(**create_parameters(Anthropic, **kwargs))
        self._async_client = AsyncAnthropic(**create_parameters(AsyncAnthropic, **kwargs))

    @override
    def generate_message(self, **kwargs) -> Message | Stream[RawMessageStreamEvent]:
        completion_params = create_parameters(self._client.messages.create, **kwargs)
        return self._client.messages.create(**completion_params)

    @override
    async def generate_message_async(
        self, **kwargs
    ) -> Message | AsyncStream[RawMessageStreamEvent]:
        completion_params = create_parameters(self._async_client.messages.create, **kwargs)
        return await self._async_client.messages.create(**completion_params)
