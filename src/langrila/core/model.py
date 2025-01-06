import inspect
from logging import Logger
from typing import Any, AsyncGenerator, Callable, Generator, Generic, Sequence

from .client import LLMClient
from .embedding import EmbeddingResults
from .logger import DEFAULT_LOGGER as default_logger
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
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        logger: Logger | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.logger = logger or default_logger
        self.system_instruction = system_instruction
        self.tools = tools or []
        self.init_kwargs = kwargs

    def _prepare_client_request(
        self,
        messages: LLMInput,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        **kwargs: Any,
    ) -> tuple[list[ClientMessage], ClientSystemMessage | None, Any]:
        messages = self._process_user_prompt(messages)

        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        if __tools := tools or self.tools:
            _tools = self._prepare_tools(__tools)
            all_kwargs["tools"] = self.client.map_to_client_tools(tools=_tools)

        messages = self._convert_message_to_list(messages)
        mapped_messages = self._response_to_prompt(messages)

        self.logger.debug(f"Prompt: {mapped_messages[-1].contents}")

        prompt = self.client.map_to_client_prompts(mapped_messages)
        _system_instruction = self.client._setup_system_instruction(system_instruction)

        return prompt, _system_instruction, all_kwargs

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
        prompt, _system_instruction, all_kwargs = self._prepare_client_request(
            messages, system_instruction or self.system_instruction, tools, **kwargs
        )

        self.logger.info("{name}: Generating text".format(name=all_kwargs.get("name", "root")))
        response = self.client.generate_text(prompt, _system_instruction, **all_kwargs)

        self.logger.debug(f"Response: {response.contents}")

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
        prompt, _system_instruction, all_kwargs = self._prepare_client_request(
            messages, system_instruction or self.system_instruction, tools, **kwargs
        )

        self.logger.info("{name}: Generating text".format(name=all_kwargs.get("name", "root")))
        response = await self.client.generate_text_async(prompt, _system_instruction, **all_kwargs)

        self.logger.debug(f"Response: {response.contents}")

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
        prompt, _system_instruction, all_kwargs = self._prepare_client_request(
            messages, system_instruction or self.system_instruction, tools, **kwargs
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
        prompt, _system_instruction, all_kwargs = self._prepare_client_request(
            messages, system_instruction or self.system_instruction, tools, **kwargs
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
        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        self.logger.info("Generating image")
        response = self.client.generate_image(prompt, **all_kwargs)

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
        all_kwargs: Any = {**self.init_kwargs, **kwargs}

        self.logger.info("Generating image")
        response = await self.client.generate_image_async(prompt, **all_kwargs)

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
        prompt, _system_instruction, all_kwargs = self._prepare_client_request(
            messages, system_instruction, None, **kwargs
        )

        self.logger.info("Generating audio")
        response = self.client.generate_audio(prompt, _system_instruction, **all_kwargs)

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
        prompt, _system_instruction, all_kwargs = self._prepare_client_request(
            messages, system_instruction, None, **kwargs
        )

        self.logger.info("Generating audio")
        response = await self.client.generate_audio_async(prompt, _system_instruction, **all_kwargs)

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
            if isinstance(_msg_or_res, Prompt):
                messages.append(_msg_or_res)
                continue

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

    def _process_user_prompt(self, prompt: LLMInput) -> list[Prompt | Response]:
        if isinstance(prompt, (Prompt, Response)):
            return [prompt]
        elif isinstance(prompt, (str, PromptType)):
            return [Prompt(role="user", contents=prompt)]
        elif isinstance(prompt, ResponseType):
            return [Response(role="assistant", contents=[prompt])]
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
            raise ValueError("Invalid prompt type. Please provide the correct prompt type.")
