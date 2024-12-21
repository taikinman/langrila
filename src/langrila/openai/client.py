import json
import os
from typing import Any, AsyncGenerator, Generator, Literal, Sequence, cast

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import Choice
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam,
)
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.shared_params.function_definition import FunctionDefinition

from ..core.client import LLMClient
from ..core.embedding import EmbeddingResults
from ..core.prompt import (
    AudioPrompt,
    ImagePrompt,
    PDFPrompt,
    Prompt,
    PromptType,
    TextPrompt,
    ToolCallPrompt,
    ToolUsePrompt,
    URIPrompt,
    VideoPrompt,
)
from ..core.response import Response, ResponseType, TextResponse, ToolCallResponse
from ..core.tool import Tool
from ..core.usage import Usage
from ..utils import create_parameters, make_batch

OpenAIMessage = (
    ChatCompletionUserMessageParam
    | ChatCompletionSystemMessageParam
    | ChatCompletionAssistantMessageParam
    | ChatCompletionToolMessageParam
    | ChatCompletionMessage
)

OpenAIMessageContentType = (
    ChatCompletionContentPartTextParam
    | ChatCompletionContentPartImageParam
    | ChatCompletionMessageToolCall
    | ChatCompletionContentPartInputAudioParam
    | ChatCompletionMessageToolCall
    | ChatCompletionToolMessageParam
)


class OpenAIClient(
    LLMClient[
        OpenAIMessage,
        ChatCompletionContentPartParam,
        ChatCompletionToolParam,
    ]
):
    """
    The wrapper client for OpenAI API.

    Parameters
    ----------
    api_key_env_name : str, optional
        Environment variable name for the API key, by default None
    api_type : Literal["openai", "azure"], optional
        API type, by default "openai"
    organization_id_env_name : str, optional
        Environment variable name for the organization ID, by default None
    kwargs : Any
        Additional keyword arguments to pass to the API client.
        Basically the same as the parameters in the raw OpenAI API.
        For more details, see the OpenAI API documentation.
    """

    def __init__(
        self,
        api_key_env_name: str | None = None,
        api_type: Literal["openai", "azure"] = "openai",
        organization_id_env_name: str | None = None,
        **kwargs: Any,
    ):
        self.api_key = os.getenv(api_key_env_name) if api_key_env_name else None
        self.organization = (
            os.getenv(organization_id_env_name) if organization_id_env_name else None
        )
        self.api_type = api_type

        if self.api_type == "openai":
            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
                **create_parameters(OpenAI, **kwargs),
            )
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                **create_parameters(AsyncOpenAI, **kwargs),
            )
        elif self.api_type == "azure":
            self._client = AzureOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                **create_parameters(AzureOpenAI, **kwargs),
            )
            self._async_client = AsyncAzureOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                **create_parameters(AsyncAzureOpenAI, **kwargs),
            )
        else:
            raise ValueError(f"Invalid API type: {self.api_type}")

    def _setup_system_instruction(
        self, system_instruction: str | Prompt | list[str | PromptType]
    ) -> list[OpenAIMessage]:
        if isinstance(system_instruction, list):
            prompt = Prompt(contents=system_instruction, role="system")
        elif isinstance(system_instruction, str):
            prompt = Prompt(contents=[system_instruction], role="system")
        else:
            prompt = system_instruction

        client_prompt = self.map_to_client_prompt(prompt)
        if isinstance(client_prompt, list):
            return client_prompt
        else:
            return [client_prompt]

    def generate_text(self, messages: list[OpenAIMessage], **kwargs: Any) -> Response:
        """
        Generate text based on the given messages.

        Parameters
        ----------
        messages : list[OpenAIMessage]
            List of messages to generate text from.
        **kwargs : Any
            Additional keyword arguments to pass to the API client.
            Basically the same as the parameters in the raw OpenAI API.
            For more details, see the OpenAI API documentation.

        Returns
        ----------
        Response
            Response object containing the generated text.
        """
        if system_instruction := kwargs.get("system_instruction"):
            system_messages = self._setup_system_instruction(system_instruction)
            messages = system_messages + messages

        if not isinstance(kwargs.get("response_format", NOT_GIVEN), (NotGiven, dict)):
            completion_params = create_parameters(
                self._client.beta.chat.completions.parse, **kwargs
            )
            response = self._client.beta.chat.completions.parse(
                messages=messages,  # type: ignore
                **completion_params,
            )
        else:
            completion_params = create_parameters(self._client.chat.completions.create, **kwargs)
            response = self._client.chat.completions.create(
                messages=messages,  # type: ignore
                **completion_params,
            )

        choices = response.choices
        usage = Usage(
            model_name=cast(str, kwargs.get("model")),
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            # raw=response.usage, # NOTE: This is not JSON serializable
        )

        contents: list[TextResponse | ToolCallResponse] = []
        for choice in choices:
            if content := choice.message.content:
                response_message = content.strip("\n")
                contents.append(TextResponse(text=response_message))
            elif tool_calls := choice.message.tool_calls:
                for tool_call in tool_calls:
                    contents.append(
                        ToolCallResponse(
                            name=tool_call.function.name,
                            call_id=tool_call.id,
                            args=tool_call.function.arguments,
                        )
                    )

        return Response(
            contents=cast(list[ResponseType], contents),
            usage=usage,
            raw=response,
            name=cast(str, kwargs.get("name")),
            prompt=messages,
        )

    async def generate_text_async(self, messages: list[OpenAIMessage], **kwargs: Any) -> Response:
        """
        Generate text based on the given messages asynchronously.

        Parameters
        ----------
        messages : list[OpenAIMessage]
            List of messages to generate text from.
        **kwargs : Any
            Additional keyword arguments to pass to the API client.
            Basically the same as the parameters in the raw OpenAI API.
            For more details, see the OpenAI API documentation.

        Returns
        ----------
        Response
            Response object containing the generated text.
        """
        if system_instruction := kwargs.get("system_instruction"):
            system_messages = self._setup_system_instruction(system_instruction)
            messages = system_messages + messages

        if not isinstance(kwargs.get("response_format", NOT_GIVEN), (NotGiven, dict)):
            completion_params = create_parameters(
                self._client.beta.chat.completions.parse, **kwargs
            )
            response = self._client.beta.chat.completions.parse(
                messages=messages,  # type: ignore
                **completion_params,
            )
        else:
            completion_params = create_parameters(self._client.chat.completions.create, **kwargs)
            response = await self._async_client.chat.completions.create(
                messages=messages,  # type: ignore
                **completion_params,
            )

        choices = response.choices
        usage = Usage(
            model_name=cast(str, kwargs.get("model")),
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            # raw=response.usage, # NOTE: This is not JSON serializable
        )

        contents: list[TextResponse | ToolCallResponse] = []
        for choice in choices:
            if content := choice.message.content:
                response_message = content.strip("\n")
                contents.append(TextResponse(text=response_message))
            elif tool_calls := choice.message.tool_calls:
                for tool_call in tool_calls:
                    contents.append(
                        ToolCallResponse(
                            name=tool_call.function.name,
                            call_id=tool_call.id,
                            args=tool_call.function.arguments,
                        )
                    )

        return Response(
            contents=cast(list[ResponseType], contents),
            usage=usage,
            raw=response,
            name=cast(str, kwargs.get("name")),
            prompt=messages,
        )

    def stream_text(
        self, messages: list[OpenAIMessage], **kwargs: Any
    ) -> Generator[Response, None, None]:
        """
        Stream text based on the given messages.

        Parameters
        ----------
        messages : list[OpenAIMessage]
            List of messages to generate text from.
        **kwargs : Any
            Additional keyword arguments to pass to the API client.
            Basically the same as the parameters in the raw OpenAI API.
            For more details, see the OpenAI API documentation.

        Yields
        ----------
        Response
            Response object containing the generated text.
        """
        if system_instruction := kwargs.get("system_instruction"):
            system_messages = self._setup_system_instruction(system_instruction)
            messages = system_messages + messages

        if not isinstance(kwargs.get("response_format", NOT_GIVEN), (NotGiven, dict)):
            raise ValueError("Streaming is not supported for OpenAI.")
        else:
            completion_params = create_parameters(self._client.chat.completions.create, **kwargs)
            response = self._client.chat.completions.create(
                messages=messages,  # type: ignore
                stream=True,
                stream_options={"include_usage": True},
                **completion_params,
            )

        chunk_texts = ""
        chunk_args = ""
        funcname = ""
        call_id = ""
        usage = Usage(model_name=cast(str | None, kwargs.get("model")))
        res: TextResponse | ToolCallResponse | None = None
        contents: list[TextResponse | ToolCallResponse] = []
        for chunk in response:
            if choices := cast(list[Choice], chunk.choices):
                delta = choices[0].delta
                if content := delta.content:
                    chunk_texts += content.strip("\n")
                    res = TextResponse(text=chunk_texts)

                    yield Response(
                        contents=[res],
                        usage=usage,
                        raw=chunk,
                        name=cast(str | None, kwargs.get("name")),
                    )
                    continue
                elif tool_calls := delta.tool_calls:
                    for tool_call in tool_calls:
                        if tool_call.id:
                            call_id = tool_call.id

                        if function := tool_call.function:
                            if function.name and funcname != function.name:
                                if res:
                                    contents.append(res)
                                    res = None
                                    chunk_args = ""

                                funcname = function.name

                            if args := function.arguments:
                                chunk_args += args

                        if funcname and chunk_args:
                            res = ToolCallResponse(name=funcname, call_id=call_id, args=chunk_args)

                            yield Response(
                                contents=[res],
                                usage=usage,
                                raw=chunk,
                                name=cast(str | None, kwargs.get("name")),
                            )
                    continue

        if res:
            contents.append(res)
            funcname = ""
            chunk_args = ""
            call_id = ""

            yield Response(
                contents=contents,
                usage=usage,
                raw=response,
                name=cast(str | None, kwargs.get("name")),
                is_last_chunk=True,
            )

        chunk_texts = ""
        res = None

    async def stream_text_async(
        self, messages: list[OpenAIMessage], **kwargs: Any
    ) -> AsyncGenerator[Response, None]:
        """
        Asynchronously stream text based on the given messages.

        Parameters
        ----------
        messages : list[OpenAIMessage]
            List of messages to generate text from.
        **kwargs : Any
            Additional keyword arguments to pass to the API client.
            Basically the same as the parameters in the raw OpenAI API.
            For more details, see the OpenAI API documentation.

        Yields
        ----------
        Response
            Response object containing the generated text.
        """

        if system_instruction := kwargs.get("system_instruction"):
            system_messages = self._setup_system_instruction(system_instruction)
            messages = system_messages + messages

        if not isinstance(kwargs.get("response_format", NOT_GIVEN), (NotGiven, dict)):
            raise ValueError("Streaming is not supported for OpenAI.")
        else:
            completion_params = create_parameters(self._client.chat.completions.create, **kwargs)
            response = await self._async_client.chat.completions.create(
                messages=messages,  # type: ignore
                stream=True,
                stream_options={"include_usage": True},
                **completion_params,
            )

        chunk_texts = ""
        chunk_args = ""
        funcname = ""
        call_id = ""
        usage = Usage(model_name=cast(str | None, kwargs.get("model")))
        res: TextResponse | ToolCallResponse | None = None
        contents: list[TextResponse | ToolCallResponse] = []
        async for chunk in response:
            if choices := cast(list[Choice], chunk.choices):
                delta = choices[0].delta
                if content := delta.content:
                    chunk_texts += content.strip("\n")
                    res = TextResponse(text=chunk_texts)

                    yield Response(
                        contents=[res],
                        usage=usage,
                        raw=chunk,
                        name=cast(str | None, kwargs.get("name")),
                    )
                    continue
                elif tool_calls := delta.tool_calls:
                    for tool_call in tool_calls:
                        if tool_call.id:
                            call_id = tool_call.id

                        if function := tool_call.function:
                            if function.name and funcname != function.name:
                                if res:
                                    contents.append(res)
                                    res = None
                                    chunk_args = ""

                                funcname = function.name

                            if args := function.arguments:
                                chunk_args += args

                        if funcname and chunk_args:
                            res = ToolCallResponse(name=funcname, call_id=call_id, args=chunk_args)

                            yield Response(
                                contents=[res],
                                usage=usage,
                                raw=chunk,
                                name=cast(str | None, kwargs.get("name")),
                            )
                    continue

        if res:
            contents.append(res)
            funcname = ""
            chunk_args = ""
            call_id = ""

        yield Response(
            contents=contents,
            usage=usage,
            raw=response,
            name=cast(str | None, kwargs.get("name")),
            is_last_chunk=True,
        )

        chunk_texts = ""
        res = None

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        """
        Embed text based on the given texts.

        Parameters
        ----------
        texts : Sequence[str]
            List of texts to embed.
        **kwargs : Any
            Additional keyword arguments to pass to the API client.
            Basically the same as the parameters in the raw OpenAI API.
            For more details, see the OpenAI API documentation.

        Returns
        ----------
        EmbeddingResults
            Embedding results object containing the embeddings.
        """
        embed_params = create_parameters(self._client.embeddings.create, **kwargs)

        embeddings = []
        if not isinstance(texts, list):
            texts = [texts]

        total_usage = Usage(model_name=kwargs.get("model"))
        batch_size = kwargs.get("batch_size", 10)
        for batch in make_batch(texts, batch_size=batch_size):
            response = self._client.embeddings.create(input=batch, **embed_params)
            embeddings.extend([e.embedding for e in response.data])
            total_usage.update(
                **{
                    "prompt_tokens": response.usage.prompt_tokens,
                }
            )

        results = EmbeddingResults(
            text=texts,
            embeddings=embeddings,
            usage=total_usage,
        )
        return results

    async def embed_text_async(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        """
        Embed text based on the given texts asynchronously.

        Parameters
        ----------
        texts : Sequence[str]
            List of texts to embed.
        **kwargs : Any
            Additional keyword arguments to pass to the API client.
            Basically the same as the parameters in the raw OpenAI API.
            For more details, see the OpenAI API documentation.

        Returns
        ----------
        EmbeddingResults
            Embedding results object containing the embeddings.
        """
        embed_params = create_parameters(self._async_client.embeddings.create, **kwargs)

        embeddings = []
        if not isinstance(texts, list):
            texts = [texts]

        total_usage = Usage(model_name=kwargs.get("model"))
        batch_size = kwargs.get("batch_size", 10)
        for batch in make_batch(texts, batch_size=batch_size):
            response = await self._async_client.embeddings.create(input=batch, **embed_params)
            embeddings.extend([e.embedding for e in response.data])
            total_usage.update(
                **{
                    "prompt_tokens": response.usage.prompt_tokens,
                }
            )

        results = EmbeddingResults(
            text=texts,
            embeddings=embeddings,
            usage=total_usage,
        )
        return results

    def map_to_client_prompt(self, message: Prompt) -> OpenAIMessage | list[OpenAIMessage]:
        """
        Map a message to a client-specific representation.

        Parameters
        ----------
        message : Prompt
            Prompt to map.

        Returns
        ----------
        OpenAIMessage | list[OpenAIMessage]
            Client-specific message representation.
        """
        tool_use = False
        contents: list[OpenAIMessageContentType] = []
        tool_use_messages: list[OpenAIMessage] = []
        tool_call_messages: list[ChatCompletionMessageToolCall] = []
        for content in message.contents:
            if isinstance(content, str):
                contents.append(ChatCompletionContentPartTextParam(text=content, type="text"))
            elif isinstance(content, TextPrompt):
                contents.append(ChatCompletionContentPartTextParam(text=content.text, type="text"))
            elif isinstance(content, ImagePrompt):
                contents.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(
                            url=f"data:image/{content.format};base64,{content.image}",
                            detail=content.resolution if content.resolution else "auto",
                        ),
                    )
                )
            elif isinstance(content, PDFPrompt):
                contents.extend(
                    [
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url=ImageURL(
                                url=f"data:image/{img.format};base64,{img.image}",
                                detail="high",
                            ),
                        )
                        for img in content.as_image_content()
                    ]
                )
            elif isinstance(content, URIPrompt):
                raise NotImplementedError
            elif isinstance(content, VideoPrompt):
                contents.extend(
                    [
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url=ImageURL(
                                url=f"data:image/{img.format};base64,{img.image}",
                                detail=content.image_resolution,
                            ),
                        )
                        for img in content.as_image_content()
                    ]
                )
            elif isinstance(content, AudioPrompt):
                raise NotImplementedError
            elif isinstance(content, ToolCallPrompt):
                content_args = content.args
                tool_call_messages.append(
                    ChatCompletionMessageToolCall(
                        id="call_" + cast(str, content.call_id).split("_")[-1],
                        type="function",
                        function=Function(arguments=json.dumps(content_args), name=content.name),
                    )
                )

            elif isinstance(content, ToolUsePrompt):
                tool_use = True
                tool_call_id = "call_" + cast(str, content.call_id).split("_")[-1]
                if content.output:
                    tool_result = content.output
                elif content.error:
                    tool_result = content.error
                else:
                    raise ValueError("Tool use prompt must have output or error.")

                tool_use_messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool", content=tool_result, tool_call_id=tool_call_id
                    )
                )

        if tool_call_messages:
            return ChatCompletionMessage(
                role="assistant",
                tool_calls=tool_call_messages,
            )
        elif tool_use:
            return tool_use_messages
        elif message.role == "assistant":
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=contents,  # type: ignore
                name=cast(str, message.name),
            )
        elif message.role == "system":
            return ChatCompletionSystemMessageParam(
                role="system",
                content=contents,  # type: ignore
                name=cast(str, message.name),
            )
        elif message.role == "developer":
            return {
                "role": "developer",
                "content": contents,  # type: ignore
                "name": cast(str, message.name),
            }
        else:
            return ChatCompletionUserMessageParam(
                role="user",
                content=contents,  # type: ignore
                name=cast(str, message.name),
            )

    def map_to_client_tools(self, tools: list[Tool]) -> list[ChatCompletionToolParam]:
        """
        Map tools to client-specific representations.

        Parameters
        ----------
        tools : list[Tool]
            List of tools to map.

        Returns
        ----------
        list[FunctionDefinition]
            List of client-specific tool representations.
        """
        return [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=cast(str, tool.name),
                    description=cast(str, tool.description),
                    parameters=cast(dict[str, Any], tool.schema_dict),
                ),
            )
            for tool in tools
        ]
