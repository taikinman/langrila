import json
import os
from typing import Any, AsyncGenerator, Generator, Literal, Sequence

from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicVertex,
)
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
from ..core.embedding import EmbeddingResults
from ..core.prompt import (
    AudioPrompt,
    ImagePrompt,
    PDFPrompt,
    Prompt,
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


class AnthropicClient(LLMClient[MessageParam, AnthropicContentType, ToolParam]):
    def __init__(
        self,
        api_key_env_name: str | None = None,
        api_type: Literal["anthropic", "bedrock", "vertexai"] = "anthropic",
        **kwargs: Any,
    ):
        self.api_key = os.getenv(api_key_env_name) if api_key_env_name else None
        self.api_type = api_type

        self._client: ClientType
        self._async_client: ClientType
        if self.api_type == "anthropic":
            self._client = Anthropic(api_key=self.api_key, **create_parameters(Anthropic, **kwargs))
            self._async_client = AsyncAnthropic(**create_parameters(AsyncAnthropic, **kwargs))
        elif self.api_type == "bedrock":
            self._client = AnthropicBedrock(**create_parameters(AnthropicBedrock, **kwargs))
            self._async_client = AsyncAnthropicBedrock(
                **create_parameters(AsyncAnthropicBedrock, **kwargs)
            )
        elif self.api_type == "vertexai":
            self._client = AnthropicVertex(**create_parameters(AnthropicVertex, **kwargs))
            self._async_client = AsyncAnthropicVertex(
                **create_parameters(AsyncAnthropicVertex, **kwargs)
            )

    def generate_text(self, messages: list[MessageParam], **kwargs: Any) -> Response:
        assert not kwargs.get("stream"), "Use stream_text or stream_text_async for streaming."
        completion_params = create_parameters(self._client.messages.create, **kwargs)
        response: Message = self._client.messages.create(
            messages=messages, stream=False, **completion_params
        )

        contents: list[TextResponse | ToolCallResponse] = []
        for content in response.content:
            if isinstance(content, TextBlock):
                contents.append(TextResponse(text=content.text))
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

    async def generate_text_async(self, messages: list[MessageParam], **kwargs: Any) -> Response:
        assert not kwargs.get("stream"), "Use stream_text or stream_text_async for streaming."
        completion_params = create_parameters(self._client.messages.create, **kwargs)
        response: Message = await self._async_client.messages.create(
            messages=messages, stream=False, **completion_params
        )

        contents: list[TextResponse | ToolCallResponse] = []
        for content in response.content:
            if isinstance(content, TextBlock):
                contents.append(TextResponse(text=content.text))
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
        self, messages: list[MessageParam], **kwargs: Any
    ) -> Generator[Response, None, None]:
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
                        res = TextResponse(text=chunk_texts)
                        yield Response(
                            contents=[res],
                            usage=usage,
                        )
                        continue

                    elif isinstance(response.content_block, ToolUseBlock):
                        funcname = response.content_block.name
                        call_id = response.content_block.id
                        chunk_args = ""
                        res = ToolCallResponse(name=funcname, call_id=call_id, args=chunk_args)
                        yield Response(contents=[res], usage=usage)
                        continue

                elif isinstance(response, RawContentBlockDeltaEvent):
                    if isinstance(response.delta, TextDelta):
                        chunk_texts += response.delta.text
                        res = TextResponse(text=chunk_texts)
                        yield Response(
                            contents=[res],
                            usage=usage,
                        )
                        continue

                    elif isinstance(response.delta, InputJSONDelta):
                        chunk_args += response.delta.partial_json

                        res = ToolCallResponse(name=funcname, call_id=call_id, args=chunk_args)
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
                    usage.update(
                        **{"output_tokens": usage.output_tokens + response.usage.output_tokens},
                    )

        yield Response(contents=contents, usage=usage)

    async def stream_text_async(
        self, messages: list[MessageParam], **kwargs: Any
    ) -> AsyncGenerator[Response, None]:
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
                        res = TextResponse(text=chunk_texts)
                        yield Response(
                            contents=[res],
                            usage=usage,
                        )
                        continue

                    elif isinstance(response.content_block, ToolUseBlock):
                        funcname = response.content_block.name
                        call_id = response.content_block.id
                        chunk_args = ""
                        res = ToolCallResponse(name=funcname, call_id=call_id, args=chunk_args)
                        yield Response(contents=[res], usage=usage)
                        continue

                elif isinstance(response, RawContentBlockDeltaEvent):
                    if isinstance(response.delta, TextDelta):
                        chunk_texts += response.delta.text
                        res = TextResponse(text=chunk_texts)
                        yield Response(
                            contents=[res],
                            usage=usage,
                        )
                        continue

                    elif isinstance(response.delta, InputJSONDelta):
                        chunk_args += response.delta.partial_json

                        res = ToolCallResponse(name=funcname, call_id=call_id, args=chunk_args)
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
                    usage.update(
                        **{"output_tokens": usage.output_tokens + response.usage.output_tokens},
                    )

        yield Response(contents=contents, usage=usage)

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

    def map_to_client_prompt(self, message: Prompt) -> MessageParam | list[MessageParam]:
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
                            media_type=f"image/{content.image.format}",
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
        list[Any]
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
