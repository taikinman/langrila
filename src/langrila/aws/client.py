import hashlib
import json
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, cast

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

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
from ..core.response import Response, ResponseType, TextResponse, ToolCallResponse
from ..core.tool import Tool
from ..core.usage import Usage
from ..utils import utf8_to_bytes

BedrockMessageType = dict[str, Any]
BedrockSystemMessageType = list[dict[str, Any]]
BedrockMessageContentType = dict[str, Any]
BedrockToolType = dict[str, Any]


class BedrockClient(
    LLMClient[
        BedrockMessageType, BedrockSystemMessageType, BedrockMessageContentType, BedrockToolType
    ]
):
    """
    Wrapper client for Amazon Bedrock Converse API.

    Parameters
    ----------
    region_name : str, optional
        The region name of the S3 bucket, by default None.
    api_version : str, optional
        The API version of the S3 client, by default None.
    use_ssl : bool, optional
        Whether to use SSL for the S3 client, by default True.
    verify : str or bool, optional
        Whether to verify the SSL certificate for the S3 client, by default None.
    endpoint_url_env_name : str, optional
        The environment variable name for the S3 endpoint URL, by default None.
    aws_access_key_env_name : str, optional
        The environment variable name for the AWS access key ID, by default None.
    aws_secret_key_env_name : str, optional
        The environment variable name for the AWS secret access key, by default None.
    aws_session_token_env_name : str, optional
        The environment variable name for the AWS session token, by default None.
    boto_config : BotoConfig, optional
        The configuration for the S3 client, by default None.
    """

    def __init__(
        self,
        region_name: str | None = None,
        api_version: str | None = None,
        use_ssl: bool = True,
        verify: str | bool | None = None,
        endpoint_url_env_name: str | None = None,
        aws_access_key_env_name: str | None = None,
        aws_secret_key_env_name: str | None = None,
        aws_session_token_env_name: str | None = None,
        boto_config: BotoConfig | None = None,
    ) -> None:
        self.region_name = region_name
        self.api_version = api_version
        self.use_ssl = use_ssl
        self.verify = verify
        self.endpoint_url = os.getenv(endpoint_url_env_name) if endpoint_url_env_name else None
        self.aws_access_key_id = (
            os.getenv(aws_access_key_env_name) if aws_access_key_env_name else None
        )
        self.aws_secret_access_key = (
            os.getenv(aws_secret_key_env_name) if aws_secret_key_env_name else None
        )
        self.aws_session_token = (
            os.getenv(aws_session_token_env_name) if aws_session_token_env_name else None
        )
        self.boto_config = boto_config

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self.region_name,
            api_version=self.api_version,
            use_ssl=self.use_ssl,
            verify=self.verify,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            config=self.boto_config,
        )

    def generate_text(
        self,
        messages: list[BedrockMessageType],
        system_instruction: BedrockSystemMessageType | None = None,
        **kwargs: Any,
    ) -> Response:
        name = cast(str | None, kwargs.pop("name", None))

        if system_instruction:
            kwargs["system"] = system_instruction

        assert not kwargs.get("stream"), "Use stream_text or stream_text_async for streaming."

        if tools := kwargs.pop("tools", []):
            kwargs["toolConfig"] = {"tools": tools}

            if tool_choice := kwargs.pop("toolChoice", None):
                kwargs["toolConfig"]["toolChoice"] = tool_choice

        response = self._client.converse(
            messages=messages,
            **kwargs,
        )

        contents: list[ResponseType] = []
        if response_message := response.get("output", {}).get("message"):
            for content in response_message["content"]:
                if text_content := content.get("text"):
                    contents.append(TextResponse(text=text_content))
                if tool_spec := content.get("toolUse"):
                    contents.append(
                        ToolCallResponse(
                            name=tool_spec["name"],
                            args=json.dumps(tool_spec["input"], ensure_ascii=False),
                            call_id=tool_spec["toolUseId"],
                        )
                    )

        response_usage = response.get("usage", {})
        usage = Usage(
            model_name=kwargs.get("modelId"),
            prompt_tokens=response_usage.get("inputTokens", 0),
            output_tokens=response_usage.get("outputTokens", 0),
        )
        return Response(
            role="assistant",
            contents=contents,
            usage=usage,
            raw=response,
            name=name,
        )

    async def generate_text_async(
        self,
        messages: list[BedrockMessageType],
        system_instruction: BedrockSystemMessageType | None = None,
        **kwargs: Any,
    ) -> Response:
        # How to implement this method?
        raise NotImplementedError

    def stream_text(
        self,
        messages: list[BedrockMessageType],
        system_instruction: BedrockSystemMessageType | None = None,
        **kwargs: Any,
    ) -> Generator[Response, None, None]:
        name = cast(str | None, kwargs.pop("name", None))

        if system_instruction:
            kwargs["system"] = system_instruction

        if tools := kwargs.pop("tools", []):
            kwargs["toolConfig"] = {"tools": tools}

            if tool_choice := kwargs.pop("toolChoice", None):
                kwargs["toolConfig"]["toolChoice"] = tool_choice

        streamed_response = self._client.converse_stream(
            messages=messages,
            **kwargs,
        )

        combined_chunk_text = ""
        chunk_args = ""
        res: TextResponse | ToolCallResponse | None = None
        contents: list[TextResponse | ToolCallResponse] = []
        usage = Usage(model_name=kwargs.get("modelId"))
        for chunk in streamed_response.get("stream", []):
            if message_start := chunk.get("messageStart"):
                role = message_start.get("role")
            elif content_block_delta := chunk.get("contentBlockDelta"):
                delta = content_block_delta.get("delta", {})

                if text := delta.get("text"):
                    combined_chunk_text += text
                    res = TextResponse(text=combined_chunk_text)
                    yield Response(
                        role=role,
                        contents=[res],
                        raw=chunk,
                        name=name,
                    )
                if tool_use := delta.get("toolUse"):
                    if tool_use.get("toolUseId"):
                        tool_use_id = tool_use["toolUseId"]
                        tool_name = tool_use["name"]
                    elif tool_use.get("input"):
                        chunk_args = tool_use["input"]

                        res = ToolCallResponse(name=tool_name, args=chunk_args, call_id=tool_use_id)
                        yield Response(
                            role=role,
                            contents=[res],
                            raw=chunk,
                            name=name,
                        )
            elif content_block_start := chunk.get("contentBlockStart"):
                start = content_block_start.get("start", {})

                if text := start.get("text"):
                    combined_chunk_text += text
                    res = TextResponse(text=combined_chunk_text)
                    yield Response(
                        role=role,
                        contents=[res],
                        raw=chunk,
                        name=name,
                    )
                if tool_use := start.get("toolUse"):
                    if tool_use.get("toolUseId"):
                        tool_use_id = tool_use["toolUseId"]
                        tool_name = tool_use["name"]
                    elif tool_use.get("input"):
                        chunk_args = tool_use["input"]

                        res = ToolCallResponse(name=tool_name, args=chunk_args, call_id=tool_use_id)
                        yield Response(
                            role=role,
                            contents=[res],
                            raw=chunk,
                            name=name,
                        )
            elif content_block_stop := chunk.get("contentBlockStop"):
                contents.append(res)

                combined_chunk_text = ""
                chunk_args = ""
                tool_use_id = ""
                tool_name = None
                res = None
            elif metadata := chunk.get("metadata"):
                if usage_metadata := metadata.get("usage"):
                    usage = Usage(
                        model_name=kwargs.get("modelId"),
                        prompt_tokens=usage_metadata.get("inputTokens", 0),
                        output_tokens=usage_metadata.get("outputTokens", 0),
                    )
            else:
                continue

        yield Response(
            role=role,
            contents=contents,
            usage=usage,
            raw=chunk,
            name=name,
            is_last_chunk=True,
        )

    def stream_text_async(
        self,
        messages: list[BedrockMessageType],
        system_instruction: BedrockSystemMessageType | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Response, None]:
        # How to implement this method?
        raise NotImplementedError

    def map_to_client_prompts(self, messages: list[Prompt]) -> list[BedrockMessageType]:
        """
        Map a message to a client-specific representation.

        Parameters
        ----------
        message : Prompt
            Prompt to map.

        Returns
        ----------
        list[BedrockMessageType]
            List of provider-specific message representation.
        """
        mapped_messages: list[BedrockMessageType] = []
        for message in messages:
            if not message.contents:
                continue

            contents: list[BedrockMessageContentType] = []
            for content in message.contents:
                if isinstance(content, str):
                    contents.append(
                        {
                            "text": content,
                        }
                    )
                elif isinstance(content, TextPrompt):
                    contents.append(
                        {
                            "text": content.text,
                        }
                    )
                elif isinstance(content, ToolCallPrompt):
                    contents.append(
                        {
                            "toolUse": {
                                "toolUseId": content.call_id,
                                "name": content.name,
                                "input": json.loads(re.sub(r'\+"', '"', content.args)),
                            }
                        }
                    )
                elif isinstance(content, ToolUsePrompt):
                    contents.append(
                        {
                            "toolResult": {
                                "toolUseId": content.call_id,
                                "content": [
                                    {
                                        "text": content.output or content.error,
                                    }
                                ],
                            }
                        }
                    )
                elif isinstance(content, ImagePrompt):
                    contents.append(
                        {
                            "image": {
                                "format": content.format,
                                "source": {
                                    "bytes": utf8_to_bytes(cast(str, content.image)),
                                },
                            }
                        }
                    )
                elif isinstance(content, VideoPrompt):
                    raise NotImplementedError
                elif isinstance(content, AudioPrompt):
                    raise NotImplementedError
                elif isinstance(content, PDFPrompt):
                    contents.append(
                        {
                            "document": {
                                "format": "pdf",
                                "name": Path(content.pdf).stem,
                                "source": {
                                    "bytes": content.as_bytes(),
                                },
                            }
                        }
                    )
                elif isinstance(content, URIPrompt):
                    raise NotImplementedError

            mapped_messages.append(
                {
                    "role": message.role,
                    "content": contents,
                }
            )

        return mapped_messages

    def map_to_client_tools(self, tools: list[Tool]) -> list[BedrockToolType]:
        """
        Map tools to client-specific representations.

        Parameters
        ----------
        tools : list[Tool]
            List of tools to map.

        Returns
        ----------
        list[BedrockToolType]
            List of client-specific tool representations.
        """
        return [self.map_to_client_tool(tool=tool) for tool in tools]

    def map_to_client_tool(self, tool: Tool) -> BedrockToolType:
        if tool.schema_dict is None:
            raise ValueError("Tool schema is required.")

        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": tool.schema_dict},
            }
        }

    def setup_system_instruction(
        self, system_instruction: SystemPrompt
    ) -> BedrockSystemMessageType:
        """
        Setup the system instruction.

        Parameters
        ----------
        system_instruction : SystemPrompt
            System instruction.

        Returns
        ----------
        BedrockSystemMessageType
            List of messages with the system instruction.
        """
        return [{"text": system_instruction.contents}]
