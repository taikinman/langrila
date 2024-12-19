import inspect
from logging import Logger
from typing import Any, AsyncGenerator, Callable, Generator, Generic, Sequence, cast

from .client import LLMClient
from .embedding import EmbeddingResults
from .logger import DEFAULT_LOGGER as default_logger
from .memory import BaseConversationMemory
from .message import Message
from .prompt import ImagePrompt, Prompt, PromptType, TextPrompt, ToolCallPrompt
from .response import ImageResponse, Response, ResponseType, TextResponse, ToolCallResponse
from .tool import Tool
from .typing import ClientMessage, ClientMessageContent, ClientTool

LLMInput = (
    Prompt
    | PromptType
    | Response
    | ResponseType
    | list[Prompt | PromptType | ResponseType | Response]
)


class LLMModel(Generic[ClientMessage, ClientMessageContent, ClientTool]):
    def __init__(
        self,
        client: LLMClient[ClientMessage, ClientMessageContent, ClientTool],
        conversation_memory: BaseConversationMemory | None = None,
        logger: Logger | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.logger = logger or default_logger
        self.conversation_memory = conversation_memory
        self.init_kwargs = kwargs

    def generate_text(self, messages: LLMInput, **kwargs: Any) -> Response:
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        if tools := all_kwargs.pop("tools", None):
            _tools = self._prepare_tools(cast(list[Callable[..., Any] | Tool], tools))
            all_kwargs["tools"] = self.client.map_to_client_tools(tools=_tools)

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)

        self.logger.info("Generating text")
        response = self.client.generate_text(prompt, **all_kwargs)

        history.append(response)
        self.store_history(history)
        return response

    async def generate_text_async(self, messages: LLMInput, **kwargs: Any) -> Response:
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        if tools := all_kwargs.pop("tools", None):
            tools = self._prepare_tools(cast(list[Callable[..., Any] | Tool], tools))
            all_kwargs["tools"] = self.client.map_to_client_tools(tools=tools)

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)

        self.logger.info("Generating text")
        response = await self.client.generate_text_async(prompt, **all_kwargs)

        history.append(response)
        self.store_history(history)

        return response

    def stream_text(self, messages: LLMInput, **kwargs: Any) -> Generator[Response, None, None]:
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        if tools := all_kwargs.pop("tools", None):
            tools = self._prepare_tools(cast(list[Callable[..., Any] | Tool], tools))
            all_kwargs["tools"] = self.client.map_to_client_tools(tools=tools)

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)

        self.logger.info("Generating text")
        streamed_response = self.client.stream_text(prompt, **all_kwargs)

        if inspect.isgenerator(streamed_response):
            for chunk in streamed_response:
                yield chunk
        else:
            raise ValueError(f"Expected a generator, but got {type(streamed_response)}")

        history.append(chunk)
        self.store_history(history)

    async def stream_text_async(
        self, messages: LLMInput, **kwargs: Any
    ) -> AsyncGenerator[Response, None]:
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        if tools := all_kwargs.pop("tools", None):
            tools = self._prepare_tools(cast(list[Callable[..., Any] | Tool], tools))
            all_kwargs["tools"] = self.client.map_to_client_tools(tools=tools)

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)

        self.logger.info("Generating text")
        streamed_response = self.client.stream_text_async(prompt, **all_kwargs)

        if inspect.isasyncgen(streamed_response):
            async for chunk in streamed_response:
                yield chunk
        else:
            raise ValueError(f"Expected a async generator, but got {type(streamed_response)}")

        history.append(chunk)
        self.store_history(history)

    def generate_image(self, prompt: str, **kwargs: Any) -> Response:
        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        self.logger.info("Generating image")
        response = self.client.generate_image(prompt, **all_kwargs)
        return response

    async def generate_image_async(self, prompt: str, **kwargs: Any) -> Response:
        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        self.logger.info("Generating image")
        response = await self.client.generate_image_async(prompt, **all_kwargs)
        return response

    def generate_video(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    async def generate_video_async(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    def generate_audio(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    async def generate_audio_async(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        self.logger.info("Embedding text")
        return self.client.embed_text(texts, **all_kwargs)

    async def embed_text_async(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        self.logger.info("Embedding text")
        return await self.client.embed_text_async(texts, **all_kwargs)

    def _convert_message_to_list(
        self, message: list[Prompt | Response] | Prompt | Response
    ) -> list[Prompt | Response]:
        if not isinstance(message, list):
            return [message]

        return message

    def _response_to_prompt(self, msg_or_res: list[Prompt | Response]) -> list[Prompt]:
        """
        Convert a list of Response objects to a list of Prompt objects.
        Prompt objects are used as it is.
        """
        messages: list[Prompt] = []
        for _msg_or_res in msg_or_res:
            contents: list[PromptType] = []

            for content in _msg_or_res.contents or []:
                if isinstance(content, TextResponse):
                    contents.append(TextPrompt(text=content.text))
                elif isinstance(content, ImageResponse):
                    contents.append(ImagePrompt(image=content.image))
                elif isinstance(content, ToolCallResponse):
                    contents.append(
                        ToolCallPrompt(
                            call_id=content.call_id,
                            name=content.name or "",
                            args=content.args or "{}",
                        )
                    )
                elif isinstance(content, PromptType) and content:
                    contents.append(content)
                else:
                    raise ValueError(f"Invalid content type: {type(content)}")

            messages.append(
                Prompt(
                    contents=contents,
                    role=_msg_or_res.role,
                    name=_msg_or_res.name,
                )
            )

        return messages

    def _prepare_tools(self, tools: list[Callable[..., Any] | Tool]) -> list[Tool]:
        outputs: list[Tool] = []
        for tool in tools:
            if isinstance(tool, Tool):
                outputs.append(tool)
            elif callable(tool):
                self.logger.debug(f"Preparing tools: {tool.__name__}")
                outputs.append(Tool(tool=tool))
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        return outputs

    def store_history(self, messages: list[Prompt | Response]) -> None:
        if self.conversation_memory is not None:
            self.conversation_memory.store(
                [
                    m.model_dump(
                        include={"role", "contents", "name", "usage", "type"}, exclude={"raw"}
                    )
                    for m in messages
                    if m.contents
                ]
            )

    def load_history(self) -> list[Prompt | Response]:
        if self.conversation_memory is not None:
            messages = self.conversation_memory.load()
            return [
                Message[eval(m["type"])].model_validate({"message": m}).message  # type: ignore[misc]
                for m in messages
            ]
        return []

    def _process_user_prompt(self, prompt: LLMInput) -> list[Prompt | Response]:
        if isinstance(prompt, (Prompt, Response)):
            return [prompt]
        elif isinstance(prompt, (str, PromptType)):
            return [Prompt(role="user", contents=prompt)]
        elif isinstance(prompt, list):
            messages = []
            for p in prompt:
                messages.extend(self._process_user_prompt(p))
            return messages
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
