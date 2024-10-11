import copy
from typing import Any, AsyncGenerator, Generator, Iterable, Mapping, Union

import httpx
from anthropic._base_client import DEFAULT_MAX_RETRIES
from anthropic._types import (
    NOT_GIVEN,
    Body,
    Headers,
    NotGiven,
    ProxiesTypes,
    Query,
    Timeout,
    Transport,
)
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    message_create_params,
)
from anthropic.types.message_param import MessageParam
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.tool_param import ToolParam
from typing_extensions import Literal

from ...base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseMessage,
)
from ...llm_wrapper import ChatWrapperModule
from ...message_content import ConversationType, InputType
from ...result import CompletionResults
from ...usage import TokenCounter, Usage
from ..claude_utils import completion, get_client
from ..message import ClaudeMessage


class AnthropicChatCoreModule(BaseChatModule):
    def __init__(
        self,
        model_name: str,
        api_type: str = "anthropic",
        api_key_env_name: str | None = None,
        auth_token_env_name: str | None = None,
        endpoint_env_name: str | httpx.URL | None = None,
        aws_secret_key_env_name: str | None = None,
        aws_access_key_env_name: str | None = None,
        aws_region_env_name: str | None = None,
        aws_session_token_env_name: str | None = None,
        gc_region_env_name: str | NotGiven = NOT_GIVEN,
        gc_project_id_env_name: str | NotGiven = NOT_GIVEN,
        gc_access_token_env_name: str | None = None,
        credentials: Any | None = None,
        timeout: float | Timeout | None | NotGiven = 60,
        max_tokens: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        transport: Transport | None = None,
        proxies: ProxiesTypes | None = None,
        connection_pool_limits: httpx.Limits | None = None,
        _strict_response_validation: bool = False,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        system_instruction: str | Iterable[TextBlockParam] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
    ):
        self.model_name = model_name
        self.api_type = api_type
        self.api_key_env_name = api_key_env_name
        self.auth_token_env_name = auth_token_env_name
        self.endpoint_env_name = endpoint_env_name
        self.aws_secret_key_env_name = aws_secret_key_env_name
        self.aws_access_key_env_name = aws_access_key_env_name
        self.aws_region_env_name = aws_region_env_name
        self.aws_session_token_env_name = aws_session_token_env_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self.transport = transport
        self.proxies = proxies
        self.connection_pool_limits = connection_pool_limits
        self._strict_response_validation = _strict_response_validation
        self.conversation_length_adjuster = conversation_length_adjuster
        self.system_instruction = system_instruction
        self.gc_region_env_name = gc_region_env_name
        self.gc_project_id_env_name = gc_project_id_env_name
        self.gc_access_token_env_name = gc_access_token_env_name
        self.credentials = credentials
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self._client = get_client(
            api_type=self.api_type,
            api_key_env_name=self.api_key_env_name,
            auth_token_env_name=self.auth_token_env_name,
            endpoint_env_name=self.endpoint_env_name,
            aws_secret_key_env_name=self.aws_secret_key_env_name,
            aws_access_key_env_name=self.aws_access_key_env_name,
            aws_region_env_name=self.aws_region_env_name,
            aws_session_token_env_name=self.aws_session_token_env_name,
            gc_region_env_name=self.gc_region_env_name,
            gc_project_id_env_name=self.gc_project_id_env_name,
            gc_access_token_env_name=self.gc_access_token_env_name,
            credentials=self.credentials,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
            http_client=self.http_client,
            transport=self.transport,
            proxies=self.proxies,
            connection_pool_limits=self.connection_pool_limits,
            _strict_response_validation=self._strict_response_validation,
        )

    def run(
        self,
        messages: Iterable[MessageParam],
        **kwargs,
    ) -> CompletionResults:
        response = self._client.generate_message(
            messages=messages,
            **kwargs,
        )

        return CompletionResults(
            message=MessageParam(role=response.role, content=response.content),
            usage=Usage(
                model_name=response.model,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            ),
            prompt=copy.deepcopy(messages),
        )

    async def arun(
        self,
        messages: Iterable[MessageParam],
        **kwargs,
    ) -> CompletionResults:
        response = await self._client.generate_message_async(
            messages=messages,
            **kwargs,
        )

        return CompletionResults(
            message=MessageParam(role=response.role, content=response.content),
            usage=Usage(
                model_name=response.model,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            ),
            prompt=copy.deepcopy(messages),
        )

    def stream(
        self,
        messages: Iterable[MessageParam],
        **kwargs,
    ) -> Generator[CompletionResults, None, None]:
        response = self._client.generate_message(
            messages=messages,
            stream=True,
            **kwargs,
        )

        all_chunks = ""
        with response as stream:
            for r in stream:
                if isinstance(r, RawMessageStartEvent):
                    usage = Usage(
                        model_name=r.message.model,
                        prompt_tokens=r.message.usage.input_tokens,
                        completion_tokens=r.message.usage.output_tokens,
                    )
                    role = r.message.role
                elif isinstance(r, RawContentBlockStartEvent):
                    all_chunks += r.content_block.text
                elif isinstance(r, RawContentBlockDeltaEvent):
                    all_chunks += r.delta.text
                    yield CompletionResults(
                        message=MessageParam(
                            role=role, content=[TextBlockParam(text=all_chunks, type="text")]
                        ),
                        usage=Usage(),
                        prompt=[{}],
                    )
                elif isinstance(r, RawMessageStopEvent):
                    pass
                elif isinstance(r, RawMessageDeltaEvent):
                    usage += Usage(
                        completion_tokens=r.usage.output_tokens,
                    )

        yield CompletionResults(
            message=MessageParam(role=role, content=[TextBlockParam(text=all_chunks, type="text")]),
            usage=usage,
            prompt=copy.deepcopy(messages),
        )

    async def astream(
        self,
        messages: Iterable[MessageParam],
        **kwargs,
    ) -> AsyncGenerator[CompletionResults, None]:
        response = await self._client.generate_message_async(
            messages=messages,
            stream=True,
            **kwargs,
        )

        all_chunks = ""
        async with response as stream:
            async for r in stream:
                if isinstance(r, RawMessageStartEvent):
                    usage = Usage(
                        model_name=r.message.model,
                        prompt_tokens=r.message.usage.input_tokens,
                        completion_tokens=r.message.usage.output_tokens,
                    )
                    role = r.message.role
                elif isinstance(r, RawContentBlockStartEvent):
                    all_chunks += r.content_block.text
                elif isinstance(r, RawContentBlockDeltaEvent):
                    all_chunks += r.delta.text
                    yield CompletionResults(
                        message=MessageParam(
                            role=role, content=[TextBlockParam(text=all_chunks, type="text")]
                        ),
                        usage=Usage(),
                        prompt=[{}],
                    )
                elif isinstance(r, RawMessageStopEvent):
                    pass
                elif isinstance(r, RawMessageDeltaEvent):
                    usage += Usage(
                        completion_tokens=r.usage.output_tokens,
                    )

            yield CompletionResults(
                message=MessageParam(
                    role=role, content=[TextBlockParam(text=all_chunks, type="text")]
                ),
                usage=usage,
                prompt=copy.deepcopy(messages),
            )


class AnthropicChatModule(ChatWrapperModule):
    def __init__(
        self,
        model_name: str,
        api_type: str = "anthropic",
        api_key_env_name: str | None = None,
        auth_token_env_name: str | None = None,
        endpoint_env_name: str | httpx.URL | None = None,
        aws_secret_key_env_name: str | None = None,
        aws_access_key_env_name: str | None = None,
        aws_region_env_name: str | None = None,
        aws_session_token_env_name: str | None = None,
        gc_region_env_name: str | NotGiven = NOT_GIVEN,
        gc_project_id_env_name: str | NotGiven = NOT_GIVEN,
        gc_access_token_env_name: str | None = None,
        credentials: Any | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = 60,
        max_tokens: int | NotGiven = 2048,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        transport: Transport | None = None,
        proxies: ProxiesTypes | None = None,
        connection_pool_limits: httpx.Limits | None = None,
        _strict_response_validation: bool = False,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        token_counter: TokenCounter | None = None,
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        **kwargs,
    ):
        self.model_name = model_name
        self.api_type = api_type
        self.api_key_env_name = api_key_env_name
        self.auth_token_env_name = auth_token_env_name
        self.endpoint_env_name = endpoint_env_name
        self.aws_secret_key_env_name = aws_secret_key_env_name
        self.aws_access_key_env_name = aws_access_key_env_name
        self.aws_region_env_name = aws_region_env_name
        self.aws_session_token_env_name = aws_session_token_env_name
        self.gc_region_env_name = gc_region_env_name
        self.gc_project_id_env_name = gc_project_id_env_name
        self.gc_access_token_env_name = gc_access_token_env_name
        self.credentials = credentials
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self.transport = transport
        self.proxies = proxies
        self.connection_pool_limits = connection_pool_limits
        self._strict_response_validation = _strict_response_validation
        self.conversation_length_adjuster = conversation_length_adjuster
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # The module to call client API
        chat_model = AnthropicChatCoreModule(
            model_name=model_name,
            api_type=api_type,
            api_key_env_name=api_key_env_name,
            auth_token_env_name=auth_token_env_name,
            endpoint_env_name=endpoint_env_name,
            aws_secret_key_env_name=aws_secret_key_env_name,
            aws_access_key_env_name=aws_access_key_env_name,
            aws_region_env_name=aws_region_env_name,
            aws_session_token_env_name=aws_session_token_env_name,
            gc_region_env_name=gc_region_env_name,
            gc_project_id_env_name=gc_project_id_env_name,
            gc_access_token_env_name=gc_access_token_env_name,
            credentials=credentials,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            transport=transport,
            proxies=proxies,
            connection_pool_limits=connection_pool_limits,
            _strict_response_validation=_strict_response_validation,
            conversation_length_adjuster=conversation_length_adjuster,
            system_instruction=system_instruction,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        super().__init__(
            chat_model=chat_model,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
        )

    def _get_client_message_type(self) -> ClaudeMessage:
        return ClaudeMessage

    def _get_generation_kwargs(self, **kwargs) -> dict[str, Any]:
        _kwargs = {}
        _kwargs["model"] = kwargs.get("model_name") or self.model_name
        _kwargs["max_tokens"] = kwargs.get("max_tokens") or self.max_tokens
        _kwargs["metadata"] = kwargs.get("metadata")
        _kwargs["stop_sequences"] = kwargs.get("stop_sequences")
        _kwargs["system"] = kwargs.get("system_instruction") or self.system_instruction
        _kwargs["temperature"] = kwargs.get("temperature") or self.temperature
        _kwargs["top_k"] = kwargs.get("top_k") or self.top_k
        _kwargs["top_p"] = kwargs.get("top_p") or self.top_p
        _kwargs["extra_headers"] = kwargs.get("extra_headers")
        _kwargs["extra_query"] = kwargs.get("extra_query")
        _kwargs["extra_body"] = kwargs.get("extra_body")
        _kwargs["timeout"] = kwargs.get("timeout") or self.timeout
        return _kwargs

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = 2048,
        metadata: message_create_params.Metadata | NotGiven = NOT_GIVEN,
        stop_sequences: list[str] | NotGiven = NOT_GIVEN,
        system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 60,
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> CompletionResults:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            metadata=metadata,
            stop_sequences=stop_sequences,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        return super().run(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = 2048,
        metadata: message_create_params.Metadata | NotGiven = NOT_GIVEN,
        stop_sequences: list[str] | NotGiven = NOT_GIVEN,
        system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 60,
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> CompletionResults:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            metadata=metadata,
            stop_sequences=stop_sequences,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        return await super().arun(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = 2048,
        metadata: message_create_params.Metadata | NotGiven = NOT_GIVEN,
        stop_sequences: list[str] | NotGiven = NOT_GIVEN,
        system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 60,
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> Generator[CompletionResults, None, None]:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            metadata=metadata,
            stop_sequences=stop_sequences,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        return super().stream(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = 2048,
        metadata: message_create_params.Metadata | NotGiven = NOT_GIVEN,
        stop_sequences: list[str] | NotGiven = NOT_GIVEN,
        system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 60,
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AsyncGenerator[CompletionResults, None]:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            metadata=metadata,
            stop_sequences=stop_sequences,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        return super().astream(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )
