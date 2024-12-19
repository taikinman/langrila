from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Generic, Sequence

from .embedding import EmbeddingResults
from .prompt import Prompt
from .response import Response
from .tool import Tool
from .typing import ClientMessage, ClientMessageContent, ClientTool


class LLMClient(ABC, Generic[ClientMessage, ClientMessageContent, ClientTool]):
    @abstractmethod
    def generate_text(self, messages: list[ClientMessage], **kwargs: Any) -> Response:
        raise NotImplementedError

    async def generate_text_async(self, messages: list[ClientMessage], **kwargs: Any) -> Response:
        raise NotImplementedError

    def stream_text(
        self, messages: list[ClientMessage], **kwargs: Any
    ) -> Generator[Response, None, None]:
        raise NotImplementedError

    def stream_text_async(
        self, messages: list[ClientMessage], **kwargs: Any
    ) -> AsyncGenerator[Response, None]:
        raise NotImplementedError

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        raise NotImplementedError

    async def embed_text_async(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        raise NotImplementedError

    def generate_image(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    async def generate_image_async(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    def generate_video(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    async def generate_video_async(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    def generate_audio(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    async def generate_audio_async(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    @abstractmethod
    def map_to_client_prompt(self, message: Prompt) -> ClientMessage | list[ClientMessage]:
        """
        Map a message to a client-specific representation.

        Parameters
        ----------
        message : Prompt
            Prompt to map.

        Returns
        ----------
        Any
            Client-specific message representation.
        """
        raise NotImplementedError

    def map_to_client_prompts(self, messages: list[Prompt]) -> list[ClientMessage]:
        mapped_messages: list[ClientMessage] = []
        for message in messages:
            if message.contents:
                client_prompt = self.map_to_client_prompt(message)
                if isinstance(client_prompt, list):
                    mapped_messages.extend(client_prompt)
                else:
                    mapped_messages.append(client_prompt)
        return mapped_messages

    def map_to_client_tools(self, tools: list[Tool]) -> list[ClientTool]:
        """
        Map tools to client-specific representations.

        Parameters
        ----------
        tools : list[Tool]
            List of tools to map.

        Returns
        ----------
        list[Any]
            List of client-specific tool representations.
        """
        raise NotImplementedError