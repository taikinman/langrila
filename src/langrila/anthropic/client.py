import json
import os
from typing import Any, AsyncGenerator, Generator, Literal

from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicVertex,
)
from anthropic._types import NOT_GIVEN, NotGiven
from anthropic.types import (
    ContentBlock,
    ImageBlockParam,
    InputJSONDelta,
    Message,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextBlockParam,
    TextDelta,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from anthropic.types.image_block_param import Source

from ..core.client import LLMClient
from ..core.prompt import (
    AudioPrompt,
    ImagePrompt,
    PDFPrompt,
    Prompt,
    SystemPrompt,
    TextPrompt,
    ToolCallPrompt,
    ToolUsePrompt,
    URIPrompt,
    VideoPrompt,
)
from ..core.response import Response, TextResponse, ToolCallResponse
from ..core.tool import Tool
from ..core.usage import Usage
from ..utils import create_parameters

ClientType = (
    Anthropic
    | AnthropicBedrock
    | AnthropicVertex
    | AsyncAnthropic
    | AsyncAnthropicBedrock
    | AsyncAnthropicVertex
)

AnthropicContentType = (
    TextBlockParam | ImageBlockParam | ToolUseBlockParam | ToolResultBlockParam | ContentBlock
)


class AnthropicClient(LLMClient[MessageParam, str, AnthropicContentType, ToolParam]):
    """
    Wrapper client for the Anthropic API.

    Parameters
    ----------
    api_key_env_name : str, optional
        Environment variable name for the API key, by default None
    api_type : Literal["anthropic", "bedrock", "vertexai"], optional
        API type, by default "anthropic"
    **kwargs : Any
        Additional parameters to pass to the client.
        Basically, any parameter that the client accepts can be passed here.
        For more details, see the documentation of the anthropic api.
    """

    def __init__(
        self,
        api_key_env_name: str | None = None,
        api_type: Literal["anthropic", "bedrock", "vertexai"] = "anthropic",
        aws_access_key_env_name: str | None = None,
        aws_secret_key_env_name: str | None = None,
        aws_region: str | None = None,
        google_cloud_project_env_name: str | NotGiven = NOT_GIVEN,
        google_cloud_region_env_name: str | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ):
        self.api_key = os.getenv(api_key_env_name) if api_key_env_name else None
        self.api_type = api_type
        self.aws_access_key = (
            os.getenv(aws_access_key_env_name) if aws_access_key_env_name else None
        )
        self.aws_secret_key = (
            os.getenv(aws_secret_key_env_name) if aws_secret_key_env_name else None
        )
        self.aws_region = aws_region
        self.google_cloud_project = (
            os.getenv(google_cloud_project_env_name) if google_cloud_project_env_name else NOT_GIVEN
        )
        self.google_cloud_region = (
            os.getenv(google_cloud_region_env_name) if google_cloud_region_env_name else NOT_GIVEN
        )

        self._client: ClientType
        self._async_client: ClientType
        if self.api_type == "anthropic":
            self._client = Anthropic(
                api_key=self.api_key,
                **create_parameters(Anthropic, **kwargs),
            )
            self._async_client = AsyncAnthropic(
                api_key=self.api_key,
                **create_parameters(AsyncAnthropic, **kwargs),
            )
        elif self.api_type == "bedrock":
            self._client = AnthropicBedrock(
                aws_access_key=self.aws_access_key,
                aws_secret_key=self.aws_secret_key,
                aws_region=self.aws_region,
                **create_parameters(AnthropicBedrock, **kwargs),
            )
            self._async_client = AsyncAnthropicBedrock(
                aws_access_key=self.aws_access_key,
                aws_secret_key=self.aws_secret_key,
                aws_region=self.aws_region,
                **create_parameters(AsyncAnthropicBedrock, **kwargs),
            )
        elif self.api_type == "vertexai":
            self._client = AnthropicVertex(
                region=self.google_cloud_region,
                project_id=self.google_cloud_project,
                **create_parameters(AnthropicVertex, **kwargs),
            )
            self._async_client = AsyncAnthropicVertex(
                region=self.google_cloud_region,
                project_id=self.google_cloud_project,
                **create_parameters(AsyncAnthropicVertex, **kwargs),
            )

    def setup_system_instruction(self, system_instruction: SystemPrompt) -> str:
        """
        Setup the system instruction.

        Parameters
        ----------
        system_instruction : SystemPrompt
            System instruction.

        Returns
        ----------
        str
            List of messages with the system instruction.
        """
        return system_instruction.contents

    def generate_text(
        self, messages: list[MessageParam], system_instruction: str | None = None, **kwargs: Any
    ) -> Response:
        """
        Generate text from a list of messages.

        Parameters
        ----------
        messages : list[MessageParam]
            List of messages to generate text from.
        system_instruction : str, optional
            System instruction, by default None.
        **kwargs : Any
            Additional parameters to pass to the client.
            Same as the parameters accepted by the client.
            For more details, see the documentation of the anthropic api.

        Returns
        ----------
        Response
            Generated text response.
        """
        if system_instruction:
            kwargs["system"] = system_instruction

        assert not kwargs.get("stream"), "Use stream_text or stream_text_async for streaming."
        completion_params = create_parameters(self._client.messages.create, **kwargs)
        response: Message = self._client.messages.create(
            messages=messages, stream=False, **completion_params
        )

        contents: list[TextResponse | ToolCallResponse] = []
        for content in response.content:
            if isinstance(content, TextBlock):
                if content.text:
                    if text := content.text.strip():
                        contents.append(TextResponse(text=text))
            elif isinstance(content, ToolUseBlock):
                contents.append(
                    ToolCallResponse(
                        call_id=content.id, name=content.name, args=json.dumps(content.input)
                    )
                )

        return Response(
            contents=contents,
            usage=Usage(
                model_name=response.model,
                prompt_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            raw=response,
            name=kwargs.get("name"),
            prompt=messages,
        )

    async def generate_text_async(
        self, messages: list[MessageParam], system_instruction: str | None = None, **kwargs: Any
    ) -> Response:
        """
        Generate text from a list of messages asynchronously.

        Parameters
        ----------
        messages : list[MessageParam]
            List of messages to generate text from.
        system_instruction : str, optional
            System instruction, by default None.
        **kwargs : Any
            Additional parameters to pass to the client.
            Same as the parameters accepted by the client.
            For more details, see the documentation of the anthropic api.

        Returns
        ----------
        Response
            Generated text response.
        """
        if system_instruction:
            kwargs["system"] = system_instruction

        assert not kwargs.get("stream"), "Use stream_text or stream_text_async for streaming."
        completion_params = create_parameters(self._client.messages.create, **kwargs)
        response: Message = await self._async_client.messages.create(
            messages=messages, stream=False, **completion_params
        )

        contents: list[TextResponse | ToolCallResponse] = []
        for content in response.content:
            if isinstance(content, TextBlock):
                if content.text:
                    if text := content.text.strip():
                        contents.append(TextResponse(text=text))
            elif isinstance(content, ToolUseBlock):
                contents.append(
                    ToolCallResponse(
                        call_id=content.id, name=content.name, args=json.dumps(content.input)
                    )
                )

        return Response(
            contents=contents,
            usage=Usage(
                model_name=response.model,
                prompt_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            raw=response,
            name=kwargs.get("name"),
            prompt=messages,
        )

    def stream_text(
        self, messages: list[MessageParam], system_instruction: str | None = None, **kwargs: Any
    ) -> Generator[Response, None, None]:
        """
        Stream text from a list of messages.

        Parameters
        ----------
        messages : list[MessageParam]
            List of messages to stream text from.
        system_instruction : str, optional
            System instruction, by default None.
        **kwargs : Any
            Additional parameters to pass to the client.
            Same as the parameters accepted by the client.
            For more details, see the documentation of the anthropic api.

        Yields
        ----------
        Response
            Streamed text response.
        """
        if system_instruction:
            kwargs["system"] = system_instruction

        completion_params = create_parameters(self._client.messages.create, **kwargs)
        streamed_response: Message = self._client.messages.create(
            messages=messages, stream=True, **completion_params
        )

        chunk_texts = ""
        chunk_args = ""
        res: TextResponse | ToolCallResponse | None = None
        contents: list[TextResponse | ToolCallResponse] = []
        with streamed_response as stream:
            for response in stream:
                if isinstance(response, RawMessageStartEvent):
                    usage = Usage(
                        model_name=kwargs.get("model"),
                        prompt_tokens=response.message.usage.input_tokens,
                        output_tokens=response.message.usage.output_tokens,
                    )
                elif isinstance(response, RawContentBlockStartEvent):
                    if isinstance(response.content_block, TextBlock):
                        chunk_texts += response.content_block.text
                        if chunk_texts and response.content_block.text:
                            res = TextResponse(text=chunk_texts.strip())
                            yield Response(
                                contents=[res],
                                usage=usage,
                            )
                        continue

                    elif isinstance(response.content_block, ToolUseBlock):
                        funcname = response.content_block.name
                        call_id = response.content_block.id
                        chunk_args = ""
                        res = ToolCallResponse(
                            name=funcname, call_id=call_id, args=chunk_args.strip()
                        )
                        # yield Response(contents=[res], usage=usage)
                        continue

                elif isinstance(response, RawContentBlockDeltaEvent):
                    if isinstance(response.delta, TextDelta):
                        chunk_texts += response.delta.text
                        if chunk_texts and response.delta.text:
                            res = TextResponse(text=chunk_texts.strip())
                            yield Response(
                                contents=[res],
                                usage=usage,
                            )
                        continue

                    elif isinstance(response.delta, InputJSONDelta):
                        chunk_args += response.delta.partial_json

                        if chunk_args and response.delta.partial_json:
                            res = ToolCallResponse(
                                name=funcname, call_id=call_id, args=chunk_args.strip()
                            )
                            yield Response(
                                contents=[res],
                                usage=usage,
                            )
                        continue
                elif isinstance(response, RawContentBlockStopEvent):
                    if res:
                        contents.append(res)
                        res = None
                elif isinstance(response, RawMessageStopEvent):
                    pass
                elif isinstance(response, RawMessageDeltaEvent):
                    usage = Usage(
                        model_name=kwargs.get("model"),
                        prompt_tokens=usage.prompt_tokens,
                        output_tokens=usage.output_tokens + response.usage.output_tokens,
                    )

        yield Response(contents=contents, usage=usage, is_last_chunk=True)

    async def stream_text_async(
        self, messages: list[MessageParam], system_instruction: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[Response, None]:
        """
        Stream text from a list of messages asynchronously.

        Parameters
        ----------
        messages : list[MessageParam]
            List of messages to stream text from.
        system_instruction : str, optional
            System instruction, by default None.
        **kwargs : Any
            Additional parameters to pass to the client.
            Same as the parameters accepted by the client.
            For more details, see the documentation of the anthropic api.

        Yields
        ----------
        Response
            Streamed text response.
        """
        if system_instruction:
            kwargs["system"] = system_instruction

        completion_params = create_parameters(self._client.messages.create, **kwargs)
        streamed_response: Message = await self._async_client.messages.create(
            messages=messages, stream=True, **completion_params
        )

        chunk_texts = ""
        chunk_args = ""
        res: TextResponse | ToolCallResponse | None = None
        contents: list[TextResponse | ToolCallResponse] = []
        async with streamed_response as stream:
            async for response in stream:
                if isinstance(response, RawMessageStartEvent):
                    usage = Usage(
                        model_name=kwargs.get("model"),
                        prompt_tokens=response.message.usage.input_tokens,
                        output_tokens=response.message.usage.output_tokens,
                    )
                elif isinstance(response, RawContentBlockStartEvent):
                    if isinstance(response.content_block, TextBlock):
                        chunk_texts += response.content_block.text
                        if chunk_texts and response.content_block.text:
                            res = TextResponse(text=chunk_texts.strip())
                            yield Response(
                                contents=[res],
                                usage=usage,
                            )
                        continue

                    elif isinstance(response.content_block, ToolUseBlock):
                        funcname = response.content_block.name
                        call_id = response.content_block.id
                        chunk_args = ""
                        res = ToolCallResponse(
                            name=funcname, call_id=call_id, args=chunk_args.strip()
                        )
                        # yield Response(contents=[res], usage=usage)
                        continue

                elif isinstance(response, RawContentBlockDeltaEvent):
                    if isinstance(response.delta, TextDelta):
                        chunk_texts += response.delta.text
                        if chunk_texts and response.delta.text:
                            res = TextResponse(text=chunk_texts.strip())
                            yield Response(
                                contents=[res],
                                usage=usage,
                            )
                        continue

                    elif isinstance(response.delta, InputJSONDelta):
                        chunk_args += response.delta.partial_json

                        if chunk_args and response.delta.partial_json:
                            res = ToolCallResponse(
                                name=funcname, call_id=call_id, args=chunk_args.strip()
                            )
                            yield Response(
                                contents=[res],
                                usage=usage,
                            )
                        continue
                elif isinstance(response, RawContentBlockStopEvent):
                    if res:
                        contents.append(res)
                        res = None
                elif isinstance(response, RawMessageStopEvent):
                    pass
                elif isinstance(response, RawMessageDeltaEvent):
                    usage = Usage(
                        model_name=kwargs.get("model"),
                        prompt_tokens=usage.prompt_tokens,
                        output_tokens=usage.output_tokens + response.usage.output_tokens,
                    )

        yield Response(contents=contents, usage=usage, is_last_chunk=True)

    def map_to_client_prompt(self, message: Prompt) -> MessageParam | list[MessageParam]:
        """
        Map a message to a client-specific representation.

        Parameters
        ----------
        message : Prompt
            Prompt to map.

        Returns
        ----------
        MessageParam | list[MessageParam]
            Client-specific message representation.
        """
        contents: list[AnthropicContentType] = []
        for content in message.contents:
            if isinstance(content, str):
                contents.append(TextBlockParam(text=content, type="text"))
            elif isinstance(content, TextPrompt):
                contents.append(TextBlockParam(text=content.text, type="text"))
            elif isinstance(content, ToolCallPrompt):
                contents.append(
                    ToolUseBlockParam(
                        id="toolu_" + content.call_id.split("_")[-1],
                        name=content.name,
                        input=json.loads(content.args),
                        type="tool_use",
                    )
                )
            elif isinstance(content, ToolUsePrompt):
                contents.append(
                    ToolResultBlockParam(
                        tool_use_id="toolu_" + content.call_id.split("_")[-1],
                        type="tool_result",
                        content=[
                            TextBlockParam(
                                text=content.output if content.output else content.error,
                                type="text",
                            )
                        ],
                        is_error=bool(content.error),
                    )
                )
            elif isinstance(content, ImagePrompt):
                contents.append(
                    ImageBlockParam(
                        source=Source(
                            data=content.image,
                            media_type=f"image/{content.format}",
                            type="base64",
                        ),
                        type="image",
                    ),
                )
            elif isinstance(content, VideoPrompt):
                contents.extend(
                    [
                        ImageBlockParam(
                            source=Source(
                                data=frame.image,
                                media_type=f"image/{frame.format}",
                                type="base64",
                            ),
                            type="image",
                        )
                        for frame in content.as_image_content()
                    ]
                )
            elif isinstance(content, AudioPrompt):
                raise NotImplementedError
            elif isinstance(content, PDFPrompt):
                contents.extend(
                    [
                        ImageBlockParam(
                            source=Source(
                                data=page.image,
                                media_type=f"image/{page.format}",
                                type="base64",
                            ),
                            type="image",
                        )
                        for page in content.as_image_content()
                    ]
                )
            elif isinstance(content, URIPrompt):
                raise NotImplementedError

        return MessageParam(
            role=message.role,
            content=contents,
        )

    def map_to_client_tools(self, tools: list[Tool]) -> list[ToolParam]:
        """
        Map tools to client-specific representations.

        Parameters
        ----------
        tools : list[Tool]
            List of tools to map.

        Returns
        ----------
        list[ToolParam]
            List of client-specific tool representations.
        """
        return [self.map_to_client_tool(tool=tool) for tool in tools]

    def map_to_client_tool(self, tool: Tool, **kwargs: Any) -> ToolParam:
        if tool.schema_dict is None:
            raise ValueError("Tool schema is required.")

        schema = tool.schema_dict
        return ToolParam(
            description=tool.description,
            input_schema=schema,
            name=tool.name,
        )
