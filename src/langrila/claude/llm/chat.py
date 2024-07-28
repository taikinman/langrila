import copy
from typing import Any, AsyncGenerator, Generator, Iterable, Mapping, Union

import httpx
from anthropic._base_client import DEFAULT_MAX_RETRIES
from anthropic._types import (
    NOT_GIVEN,
    NotGiven,
    ProxiesTypes,
    Timeout,
    Transport,
)
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from anthropic.types.message_param import MessageParam
from anthropic.types.text_block_param import TextBlockParam

from ...base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseMessage,
)
from ...llm_wrapper import ChatWrapperModule
from ...result import CompletionResults
from ...usage import TokenCounter, Usage
from ..claude_utils import acompletion, completion, get_async_client, get_client
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
        timeout: Union[float, Timeout, None, NotGiven] = 600,
        max_tokens: int = 2048,
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

    def run(self, messages: list[dict[str, Any]]) -> CompletionResults:
        client = get_client(
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

        response = completion(
            client=client,
            model_name=self.model_name,
            messages=messages,
            system_instruction=self.system_instruction,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
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

    async def arun(self, messages: list[dict[str, Any]]) -> CompletionResults:
        client = get_async_client(
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

        response = await acompletion(
            client=client,
            model_name=self.model_name,
            messages=messages,
            system_instruction=self.system_instruction,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
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

    def stream(self, messages: list[dict[str, Any]]) -> Generator[CompletionResults, None, None]:
        client = get_client(
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

        response = completion(
            client=client,
            model_name=self.model_name,
            messages=messages,
            system_instruction=self.system_instruction,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stream=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
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
        self, messages: list[dict[str, Any]]
    ) -> AsyncGenerator[CompletionResults, None]:
        client = get_async_client(
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

        response = await acompletion(
            client=client,
            model_name=self.model_name,
            messages=messages,
            system_instruction=self.system_instruction,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stream=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
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
        timeout: Union[float, Timeout, None, NotGiven] = 600,
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
    ):
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
