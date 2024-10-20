from typing import Any, AsyncGenerator, Callable, Generator, Literal, Mapping

import httpx
from openai._types import NOT_GIVEN, NotGiven
from openai.lib.azure import AzureADTokenProvider
from pydantic import BaseModel

from ..base import BaseConversationLengthAdjuster, BaseConversationMemory, BaseFilter
from ..base_assembly import BaseAssembly
from ..message_content import ConversationType, InputType, Message, ToolContent
from ..result import CompletionResults, FunctionCallingResults
from ..tools import ToolConfig
from ..usage import TokenCounter, Usage
from .llm.chat import OpenAIChatModule
from .llm.function_calling import OpenAIFunctionCallingModule


class OpenAIFunctionalChat(BaseAssembly):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str | None = None,
        api_type: str = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        organization_id_env_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        timeout: int | NotGiven = NOT_GIVEN,
        max_retries: int = 2,
        seed: int | NotGiven = NOT_GIVEN,
        json_mode: bool = False,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        system_instruction: str | None = None,
        token_counter: TokenCounter | None = None,
        top_p: float | None | NotGiven = NOT_GIVEN,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        response_schema: BaseModel | None = None,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        n_results: int | NotGiven = NOT_GIVEN,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: str | Literal["auto", "none"] = "auto",
        tool_only: bool = False,
        stream_options: dict[str, Any] | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AzureADTokenProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
    ) -> None:
        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.tools = tools
        self.tool_configs = tool_configs
        self.tool_choice = tool_choice
        self.tool_only = tool_only
        self.organization_id_env_name = organization_id_env_name
        self.api_type = api_type
        self.api_version = api_version
        self.endpoint_env_name = endpoint_env_name
        self.deployment_id_env_name = deployment_id_env_name
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.seed = seed
        self.conversation_memory = conversation_memory
        self.content_filter = content_filter
        self.system_instruction = system_instruction
        self.conversation_length_adjuster = conversation_length_adjuster
        self.token_counter = token_counter
        self.json_mode = json_mode
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.user = user
        self.response_schema = response_schema
        self.stop = stop
        self.n_results = n_results
        self.json_mode = json_mode
        self.stream_options = stream_options
        self.project = project
        self.base_url = base_url
        self.azure_ad_token = azure_ad_token
        self.azure_ad_token_provider = azure_ad_token_provider
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self._strict_response_validation = _strict_response_validation

        self.chat = OpenAIChatModule(
            api_key_env_name=api_key_env_name,
            model_name=model_name,
            organization_id_env_name=organization_id_env_name,
            api_type=api_type,
            api_version=api_version,
            endpoint_env_name=endpoint_env_name,
            deployment_id_env_name=deployment_id_env_name,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            system_instruction=system_instruction,
            conversation_length_adjuster=conversation_length_adjuster,
            token_counter=token_counter,
            json_mode=json_mode,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            response_schema=response_schema,
            stop=stop,
            n_results=n_results,
            stream_options=stream_options,
            project=project,
            base_url=base_url,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )

        self.function_calling = OpenAIFunctionCallingModule(
            api_key_env_name=api_key_env_name,
            model_name=model_name,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            organization_id_env_name=organization_id_env_name,
            api_type=api_type,
            api_version=api_version,
            endpoint_env_name=endpoint_env_name,
            deployment_id_env_name=deployment_id_env_name,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            system_instruction=system_instruction,
            conversation_length_adjuster=conversation_length_adjuster,
            token_counter=token_counter,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            stop=stop,
            project=project,
            base_url=base_url,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )

    def _get_generation_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        _kwargs = {}
        _kwargs["system_instruction"] = kwargs.get("system_instruction") or self.system_instruction
        _kwargs["model_name"] = kwargs.get("model_name") or self.model_name
        _kwargs["temperature"] = kwargs.get("temperature") or self.temperature
        _kwargs["top_p"] = kwargs.get("top_p") or self.top_p
        _kwargs["stop"] = kwargs.get("stop") or self.stop
        _kwargs["frequency_penalty"] = kwargs.get("frequency_penalty ") or self.frequency_penalty
        _kwargs["presence_penalty"] = kwargs.get("presence_penalty") or self.presence_penalty
        _kwargs["user"] = kwargs.get("user") or self.user
        _kwargs["seed"] = kwargs.get("seed") or self.seed

        _kwargs["max_tokens"] = kwargs.get("max_tokens") or self.max_tokens
        _kwargs["max_completion_tokens"] = (
            kwargs.get("max_completion_tokens") or self.max_completion_tokens
        )
        _kwargs["json_mode"] = kwargs.get("json_mode") or self.json_mode
        _kwargs["response_schema"] = kwargs.get("response_schema") or self.response_schema
        _kwargs["parallel_tool_calls"] = kwargs.get("parallel_tool_calls")
        _kwargs["n_results"] = kwargs.get("n_results") or self.n_results
        _kwargs["tool_choice"] = kwargs.get("tool_choice") or self.tool_choice
        _kwargs["tools"] = kwargs.get("tools") or self.tools
        _kwargs["tool_configs"] = kwargs.get("tool_configs") or self.tool_configs
        _kwargs["stream_options"] = kwargs.get("stream_options") or self.stream_options
        _kwargs["conversation_length_adjuster"] = (
            kwargs.get("conversation_length_adjuster") or self.conversation_length_adjuster
        )

        return _kwargs

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        tool_choice: str | Literal["auto", "none"] = "auto",
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_only: bool = False,
        n_results: int | NotGiven = NOT_GIVEN,
        model_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        stop: str | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        system_instruction: str | None = None,
        seed: int | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        json_mode: bool = False,
        response_schema: BaseModel | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
    ) -> CompletionResults | FunctionCallingResults:
        _conversation_memory = self._setup_memory(conversation_memory or self.conversation_memory)

        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            user=user,
            seed=seed,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            n_results=n_results,
            parallel_tool_calls=parallel_tool_calls,
            json_mode=json_mode,
            response_schema=response_schema,
            conversation_length_adjuster=conversation_length_adjuster,
        )

        if not generation_kwargs.get("model_name"):
            raise ValueError("model_name must be provided")

        total_usage = Usage(model_name=generation_kwargs.get("model_name"))

        if generation_kwargs.get("tools"):
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt=prompt,
                init_conversation=init_conversation,
                conversation_memory=_conversation_memory,
                content_filter=content_filter,
                **generation_kwargs,
            )

            if tool_only or self.tool_only:
                self._clear_memory(_conversation_memory)

                return response_function_calling

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    name=None,
                    content=[
                        ToolContent(**content.model_dump())
                        for result in response_function_calling.results
                        for content in result.content
                    ],
                )
                init_conversation = (
                    None  # if tool is used, init_conversation is stored in the memory
                )

            total_usage += response_function_calling.usage

        response_chat: CompletionResults = self.chat.run(
            prompt=prompt,
            init_conversation=init_conversation,
            conversation_memory=_conversation_memory,
            content_filter=content_filter,
            **generation_kwargs,
        )

        total_usage += response_chat.usage
        response_chat.usage = total_usage

        self._clear_memory(_conversation_memory)

        return response_chat

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        tool_choice: str | Literal["auto", "none"] = "auto",
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_only: bool = False,
        n_results: int | NotGiven = NOT_GIVEN,
        model_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        stop: str | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        system_instruction: str | None = None,
        seed: int | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        json_mode: bool = False,
        response_schema: BaseModel | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
    ) -> CompletionResults | FunctionCallingResults:
        _conversation_memory = self._setup_memory(conversation_memory or self.conversation_memory)

        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            user=user,
            seed=seed,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            n_results=n_results,
            parallel_tool_calls=parallel_tool_calls,
            json_mode=json_mode,
            response_schema=response_schema,
            conversation_length_adjuster=conversation_length_adjuster,
        )

        if not generation_kwargs.get("model_name"):
            raise ValueError("model_name must be provided")

        total_usage = Usage(model_name=generation_kwargs.get("model_name"))

        if generation_kwargs.get("tools"):
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt=prompt,
                init_conversation=init_conversation,
                conversation_memory=_conversation_memory,
                content_filter=content_filter,
                **generation_kwargs,
            )

            if tool_only or self.tool_only:
                self._clear_memory(_conversation_memory)

                return response_function_calling

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    name=None,
                    content=[
                        ToolContent(**content.model_dump())
                        for result in response_function_calling.results
                        for content in result.content
                    ],
                )
                init_conversation = (
                    None  # if tool is used, init_conversation is stored in the memory
                )

            total_usage += response_function_calling.usage

        response_chat: CompletionResults = await self.chat.arun(
            prompt=prompt,
            init_conversation=init_conversation,
            conversation_memory=_conversation_memory,
            content_filter=content_filter,
            **generation_kwargs,
        )

        total_usage += response_chat.usage
        response_chat.usage = total_usage

        self._clear_memory(_conversation_memory)

        return response_chat

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        tool_choice: str | Literal["auto", "none"] = "auto",
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        stop: str | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        stream_options: dict[str, Any] | None = None,
        system_instruction: str | None = None,
        seed: int | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        json_mode: bool = False,
        response_schema: BaseModel | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
    ) -> Generator[CompletionResults, None, None]:
        _conversation_memory = self._setup_memory(conversation_memory or self.conversation_memory)

        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            user=user,
            seed=seed,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            json_mode=json_mode,
            response_schema=response_schema,
            stream_options=stream_options,
            conversation_length_adjuster=conversation_length_adjuster,
        )

        if not generation_kwargs.get("model_name"):
            raise ValueError("model_name must be provided")

        total_usage = Usage(model_name=generation_kwargs.get("model_name"))

        if generation_kwargs.get("tools"):
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt=prompt,
                init_conversation=init_conversation,
                conversation_memory=_conversation_memory,
                content_filter=content_filter,
                **generation_kwargs,
            )

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    name=None,
                    content=[
                        ToolContent(**content.model_dump())
                        for result in response_function_calling.results
                        for content in result.content
                    ],
                )
                init_conversation = (
                    None  # if tool is used, init_conversation is stored in the memory
                )

            total_usage += response_function_calling.usage

        response_chat = self.chat.stream(
            prompt=prompt,
            init_conversation=init_conversation,
            conversation_memory=_conversation_memory,
            content_filter=content_filter,
            **generation_kwargs,
        )

        for result in response_chat:
            if result.usage.total_tokens > 0:
                total_usage += result.usage
                result.usage = total_usage
            yield result

        self._clear_memory(_conversation_memory)

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        tool_choice: str | Literal["auto", "none"] = "auto",
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        stop: str | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        stream_options: dict[str, Any] | None = None,
        system_instruction: str | None = None,
        seed: int | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        json_mode: bool = False,
        response_schema: BaseModel | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
    ) -> AsyncGenerator[CompletionResults, None]:
        _conversation_memory = self._setup_memory(conversation_memory or self.conversation_memory)

        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            user=user,
            seed=seed,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            json_mode=json_mode,
            response_schema=response_schema,
            stream_options=stream_options,
            conversation_length_adjuster=conversation_length_adjuster,
        )

        if not generation_kwargs.get("model_name"):
            raise ValueError("model_name must be provided")

        total_usage = Usage(model_name=generation_kwargs.get("model_name"))

        if generation_kwargs.get("tools"):
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt=prompt,
                init_conversation=init_conversation,
                conversation_memory=_conversation_memory,
                content_filter=content_filter,
                **generation_kwargs,
            )

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    name=None,
                    content=[
                        ToolContent(**content.model_dump())
                        for result in response_function_calling.results
                        for content in result.content
                    ],
                )
                init_conversation = (
                    None  # if tool is used, init_conversation is stored in the memory
                )

            total_usage += response_function_calling.usage

        response_chat = await self.chat.astream(
            prompt=prompt,
            init_conversation=init_conversation,
            conversation_memory=_conversation_memory,
            content_filter=content_filter,
            **generation_kwargs,
        )

        async for result in response_chat:
            if result.usage.total_tokens > 0:
                total_usage += result.usage
                result.usage = total_usage
            yield result

        self._clear_memory(_conversation_memory)
