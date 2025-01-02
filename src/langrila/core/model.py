import inspect
from logging import Logger
from typing import Any, AsyncGenerator, Callable, Generator, Generic, Sequence

from .client import LLMClient
from .embedding import EmbeddingResults
from .logger import DEFAULT_LOGGER as default_logger
from .memory import BaseConversationMemory
from .message import Message
from .prompt import (
    AudioPrompt,
    ImagePrompt,
    Prompt,
    PromptType,
    SystemPrompt,
    TextPrompt,
    ToolCallPrompt,
)
from .response import (
    AudioResponse,
    ImageResponse,
    Response,
    ResponseType,
    TextResponse,
    ToolCallResponse,
)
from .tool import Tool
from .typing import ClientMessage, ClientMessageContent, ClientSystemMessage, ClientTool

LLMInput = (
    str
    | Prompt
    | PromptType
    | Response
    | ResponseType
    | list[str | Prompt | PromptType | ResponseType | Response]
)


class LLMModel(Generic[ClientMessage, ClientSystemMessage, ClientMessageContent, ClientTool]):
    """
    A high-level interface to interact with a language model client.

    Parameters
    ----------
    client : LLMClient
        Client to interact with.
    conversation_memory : BaseConversationMemory, optional
        Conversation memory to store and load conversation history, by default None.
    system_instruction : SystemPrompt, optional
        System instruction to generate text, by default None.
    tools : list[Callable[..., Any] | Tool], optional
        Tools that agent can use, by default None.
    logger : Logger, optional
        Logger to use, by default None.
    **kwargs : Any
        Additional arguments to pass to the client.
        What arguments are available depends on the client.
    """

    def __init__(
        self,
        client: LLMClient[ClientMessage, ClientSystemMessage, ClientMessageContent, ClientTool],
        conversation_memory: BaseConversationMemory | None = None,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        logger: Logger | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.logger = logger or default_logger
        self.conversation_memory = conversation_memory
        self.system_instruction = system_instruction
        self.tools = tools or []
        self.init_kwargs = kwargs

        if system_instruction:
            self.init_kwargs["system_instruction"] = system_instruction

    def generate_text(
        self,
        messages: LLMInput,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Generate text based on the given prompt.

        Parameters
        ----------
        messages : LLMInput
            Prompt to generate text from.
        system_instruction : SystemPrompt | None, optional
            System instruction to generate text, by default None.
        tools : list[Callable[..., Any] | Tool] | None, optional
            Tools that agent can use, by default None.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Returns
        ----------
        Response
            Generated text.
        """
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        if __tools := tools or self.tools:
            _tools = self._prepare_tools(__tools)
            all_kwargs["tools"] = self.client.map_to_client_tools(tools=_tools)

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug(f"Prompt: {mapped_messages[-1].contents}")

        # self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)
        _system_instruction = self.client._setup_system_instruction(
            system_instruction or self.system_instruction
        )

        self.logger.info("{name}: Generating text".format(name=all_kwargs.get("name", "root")))
        response = self.client.generate_text(prompt, _system_instruction, **all_kwargs)

        self.logger.debug(f"Response: {response.contents}")

        history.append(response)
        self.store_history(history)
        return response

    async def generate_text_async(
        self,
        messages: LLMInput,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Generate text based on the given prompt asynchronously.

        Parameters
        ----------
        messages : LLMInput
            Prompt to generate text from.
        system_instruction : SystemPrompt | None, optional
            System instruction to generate text, by default None.
        tools : list[Callable[..., Any] | Tool] | None, optional
            Tools that agent can use, by default None.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Returns
        ----------
        Response
            Generated text.
        """
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        if __tools := tools or self.tools:
            _tools = self._prepare_tools(__tools)
            all_kwargs["tools"] = self.client.map_to_client_tools(tools=_tools)

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug(f"Prompt: {mapped_messages[-1].contents}")

        # self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)
        _system_instruction = self.client._setup_system_instruction(
            system_instruction or self.system_instruction
        )

        self.logger.info("{name}: Generating text".format(name=all_kwargs.get("name", "root")))
        response = await self.client.generate_text_async(prompt, _system_instruction, **all_kwargs)

        self.logger.debug(f"Response: {response.contents}")

        history.append(response)
        self.store_history(history)

        return response

    def stream_text(
        self,
        messages: LLMInput,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        **kwargs: Any,
    ) -> Generator[Response, None, None]:
        """
        Stream text based on the given prompt.

        Parameters
        ----------
        messages : LLMInput
            Prompt to generate text from.
        system_instruction : SystemPrompt | None, optional
            System instruction to generate text, by default None.
        tools : list[Callable[..., Any] | Tool] | None, optional
            Tools that agent can use, by default None.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Yields
        ----------
        Response
            Generated text.
        """
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        if __tools := tools or self.tools:
            _tools = self._prepare_tools(__tools)
            all_kwargs["tools"] = self.client.map_to_client_tools(tools=_tools)

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug(f"Prompt: {mapped_messages[-1].contents}")

        # self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)
        _system_instruction = self.client._setup_system_instruction(
            system_instruction or self.system_instruction
        )

        self.logger.info("{name}: Generating text".format(name=all_kwargs.get("name", "root")))
        streamed_response = self.client.stream_text(prompt, _system_instruction, **all_kwargs)

        if inspect.isgenerator(streamed_response):
            for chunk in streamed_response:
                if chunk.is_last_chunk:
                    self.logger.debug(f"Response: {chunk.contents}")

                yield chunk
        else:
            raise ValueError(f"Expected a generator, but got {type(streamed_response)}")

        history.append(chunk)
        self.store_history(history)

    async def stream_text_async(
        self,
        messages: LLMInput,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Response, None]:
        """
        Stream text based on the given prompt asynchronously.

        Parameters
        ----------
        messages : LLMInput
            Prompt to generate text from.
        system_instruction : SystemPrompt | None, optional
            System instruction to generate text, by default None.
        tools : list[Callable[..., Any] | Tool] | None, optional
            Tools that agent can use, by default None.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Yields
        ----------
        Response
            Generated text.
        """
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        if __tools := tools or self.tools:
            _tools = self._prepare_tools(__tools)
            all_kwargs["tools"] = self.client.map_to_client_tools(tools=_tools)

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug(f"Prompt: {mapped_messages[-1].contents}")

        # self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)
        _system_instruction = self.client._setup_system_instruction(
            system_instruction or self.system_instruction
        )

        self.logger.info("{name}: Generating text".format(name=all_kwargs.get("name", "root")))
        streamed_response = self.client.stream_text_async(prompt, _system_instruction, **all_kwargs)

        if inspect.isasyncgen(streamed_response):
            async for chunk in streamed_response:
                if chunk.is_last_chunk:
                    self.logger.debug(f"Response: {chunk.contents}")

                yield chunk
        else:
            raise ValueError(f"Expected a async generator, but got {type(streamed_response)}")

        history.append(chunk)
        self.store_history(history)

    def generate_image(self, prompt: str, **kwargs: Any) -> Response:
        """
        Generate an image based on the given prompt.

        Parameters
        ----------
        prompt : str
            Prompt to generate the image from.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Returns
        ----------
        Response
            Generated image.
        """
        history = self.load_history()
        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        self.logger.info("Generating image")
        response = self.client.generate_image(prompt, **all_kwargs)

        history.append(response)
        self.store_history(history)

        return response

    async def generate_image_async(self, prompt: str, **kwargs: Any) -> Response:
        """
        Generate an image based on the given prompt asynchronously.

        Parameters
        ----------
        prompt : str
            Prompt to generate the image from.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Returns
        ----------
        Response
            Generated image.
        """
        history = self.load_history()
        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        self.logger.info("Generating image")
        response = await self.client.generate_image_async(prompt, **all_kwargs)

        history.append(response)
        self.store_history(history)

        return response

    def generate_video(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    async def generate_video_async(self, prompt: str, **kwargs: Any) -> Response:
        raise NotImplementedError

    def generate_audio(
        self,
        messages: LLMInput,
        system_instruction: SystemPrompt | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Generate audio based on the given prompt.

        Parameters
        ----------
        messages : LLMInput
            Prompt to generate audio from.
        system_instruction : SystemPrompt | None, optional
            System instruction to generate audio, by default None.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Returns
        ----------
        Response
            Generated audio.
        """
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug(f"Prompt: {mapped_messages[-1].contents}")

        # self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)
        _system_instruction = self.client._setup_system_instruction(
            system_instruction or self.system_instruction
        )

        self.logger.info("Generating audio")
        response = self.client.generate_audio(prompt, _system_instruction, **all_kwargs)

        history.append(response)
        self.store_history(history)

        return response

    async def generate_audio_async(
        self,
        messages: LLMInput,
        system_instruction: SystemPrompt | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Generate audio based on the given prompt asynchronously.

        Parameters
        ----------
        messages : LLMInput
            Prompt to generate audio from.
        system_instruction : SystemPrompt | None, optional
            System instruction to generate audio, by default None.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Returns
        ----------
        Response
            Generated audio.
        """
        history = self.load_history()
        messages = self._process_user_prompt(messages)
        history = history + messages

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        history = self._convert_message_to_list(history)
        mapped_messages = self._response_to_prompt(history)

        self.logger.debug(f"Prompt: {mapped_messages[-1].contents}")

        # self.logger.debug("Mapping Prompt to client-specific representation")
        prompt = self.client.map_to_client_prompts(mapped_messages)
        _system_instruction = self.client._setup_system_instruction(
            system_instruction or self.system_instruction
        )

        self.logger.info("Generating audio")
        response = await self.client.generate_audio_async(prompt, _system_instruction, **all_kwargs)

        history.append(response)
        self.store_history(history)

        return response

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        """
        Embed text into a vector representation.

        Parameters
        ----------
        texts : Sequence[str]
            Texts to embed.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Returns
        ----------
        EmbeddingResults
            Embedding results.
        """
        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        self.logger.info("Embedding text")
        return self.client.embed_text(texts, **all_kwargs)

    async def embed_text_async(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        """
        Embed text into a vector representation asynchronously.

        Parameters
        ----------
        texts : Sequence[str]
            Texts to embed.
        **kwargs : Any
            Additional arguments to pass to the client.
            What arguments are available depends on the client.

        Returns
        ----------
        EmbeddingResults
            Embedding results.
        """
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
                    contents.append(
                        ImagePrompt(image=content.image, format=content.format or "jpeg")
                    )
                elif isinstance(content, AudioResponse):
                    contents.append(
                        AudioPrompt(
                            audio=content.audio,
                            mime_type=content.mime_type,
                            audio_id=content.audio_id,
                        )
                    )

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
                # self.logger.debug(f"Preparing tools: {tool.__name__}")
                outputs.append(Tool(tool=tool))
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        return outputs

    def store_history(self, messages: list[Prompt | Response]) -> None:
        if self.conversation_memory is not None:
            self.conversation_memory.store(
                [
                    m.model_dump(
                        include={"role", "contents", "name", "type"}, exclude={"raw", "usage"}
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
            include_prompt_or_response = False
            include_content = False

            for p in prompt:
                if isinstance(p, (Prompt, Response)):
                    include_prompt_or_response = True
                elif isinstance(p, (str, PromptType)):
                    include_content = True

            if include_prompt_or_response and include_content:
                raise ValueError(
                    "Prompt types or roles are ambiguous. Don't mix Prompt/Response and str/content."
                )

            if include_prompt_or_response:
                messages = prompt
            elif include_content:
                messages = [Prompt(role="user", contents=prompt)]
            else:
                raise ValueError("Invalid prompt type")

            return messages

        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
