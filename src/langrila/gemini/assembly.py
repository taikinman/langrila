from typing import Any, AsyncGenerator, Callable, Generator, Iterable, Literal, Mapping, Sequence

from google.auth import credentials as auth_credentials

from ..base import (
    BaseConversationMemory,
    BaseFilter,
)
from ..base_assembly import BaseAssembly
from ..message_content import ConversationType, InputType, Message, ToolContent
from ..result import CompletionResults, FunctionCallingResults
from ..tools import ToolConfig
from ..usage import TokenCounter, Usage
from .llm.chat import GeminiChatModule
from .llm.function_calling import GeminiFunctionCallingModule


class GeminiFunctionalChat(BaseAssembly):
    def __init__(
        self,
        model_name: str | None = None,
        api_key_env_name: str | None = None,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
        timeout: int | None = None,
        content_filter: BaseFilter | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        system_instruction: str | None = None,
        token_counter: TokenCounter | None = None,
        api_type: str = "genai",
        project_id_env_name: str | None = None,
        location_env_name: str | None = None,
        experiment: str | None = None,
        experiment_description: str | None = None,
        experiment_tensorboard: str | bool | None = None,
        staging_bucket: str | None = None,
        credentials: auth_credentials.Credentials | None = None,
        encryption_spec_key_name: str | None = None,
        network: str | None = None,
        service_account: str | None = None,
        endpoint_env_name: str | None = None,
        request_metadata: Sequence[tuple[str, str]] | None = None,
        response_schema: dict[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        response_mime_type: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        n_results: int | None = None,
        tool_choice: str = "auto",
    ):
        super().__init__(conversation_memory=conversation_memory)

        self.model_name = model_name
        self.api_key_env_name = api_key_env_name
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        self.timeout = timeout
        self.content_filter = content_filter
        self.system_instruction = system_instruction
        self.token_counter = token_counter
        self.api_type = api_type
        self.project_id_env_name = project_id_env_name
        self.location_env_name = location_env_name
        self.experiment = experiment
        self.experiment_description = experiment_description
        self.experiment_tensorboard = experiment_tensorboard
        self.staging_bucket = staging_bucket
        self.credentials = credentials
        self.encryption_spec_key_name = encryption_spec_key_name
        self.network = network
        self.service_account = service_account
        self.endpoint_env_name = endpoint_env_name
        self.request_metadata = request_metadata
        self.tools = tools
        self.tool_configs = tool_configs
        self.response_mime_type = response_mime_type
        self.response_schema = response_schema
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.routing_config = routing_config
        self.logprobs = logprobs
        self.response_logprobs = response_logprobs
        self.stop_sequences = stop_sequences
        self.n_results = n_results
        self.tool_choice = tool_choice

        self.chat = GeminiChatModule(
            model_name=model_name,
            api_key_env_name=api_key_env_name,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
            timeout=timeout,
            content_filter=content_filter,
            conversation_memory=self.conversation_memory,
            system_instruction=system_instruction,
            token_counter=token_counter,
            api_type=api_type,
            project_id_env_name=project_id_env_name,
            location_env_name=location_env_name,
            experiment=experiment,
            experiment_description=experiment_description,
            experiment_tensorboard=experiment_tensorboard,
            staging_bucket=staging_bucket,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            network=network,
            service_account=service_account,
            endpoint_env_name=endpoint_env_name,
            request_metadata=request_metadata,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            n_results=n_results,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            stop_sequences=stop_sequences,
            response_mime_type=response_mime_type,
        )

        self.function_calling = GeminiFunctionCallingModule(
            model_name=model_name,
            api_key_env_name=api_key_env_name,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
            timeout=timeout,
            content_filter=content_filter,
            conversation_memory=self.conversation_memory,
            system_instruction=system_instruction,
            token_counter=token_counter,
            api_type=api_type,
            project_id_env_name=project_id_env_name,
            location_env_name=location_env_name,
            experiment=experiment,
            experiment_description=experiment_description,
            experiment_tensorboard=experiment_tensorboard,
            staging_bucket=staging_bucket,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            network=network,
            service_account=service_account,
            endpoint_env_name=endpoint_env_name,
            request_metadata=request_metadata,
            tools=tools,
            tool_configs=tool_configs,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            tool_choice=tool_choice,
            stop_sequences=stop_sequences,
        )

    def _get_generation_kwargs(self, **kwargs) -> dict[str, Any]:
        _kwargs = {}
        _kwargs["system_instruction"] = kwargs.get("system_instruction") or self.system_instruction
        _kwargs["model_name"] = kwargs.get("model_name") or self.model_name
        _kwargs["stop_sequences"] = kwargs.get("stop_sequences")
        _kwargs["max_output_tokens"] = kwargs.get("max_output_tokens") or self.max_output_tokens
        _kwargs["temperature"] = kwargs.get("temperature") or self.temperature
        _kwargs["top_p"] = kwargs.get("top_p") or self.top_p
        _kwargs["top_k"] = kwargs.get("top_k") or self.top_k
        _kwargs["presence_penalty"] = kwargs.get("presence_penalty") or self.presence_penalty
        _kwargs["frequency_penalty"] = kwargs.get("frequency_penalty") or self.frequency_penalty
        _kwargs["seed"] = kwargs.get("seed") or self.seed
        _kwargs["routing_config"] = kwargs.get("routing_config") or self.routing_config
        _kwargs["logprobs"] = kwargs.get("logprobs") or self.logprobs
        _kwargs["response_logprobs"] = kwargs.get("response_logprobs") or self.response_logprobs
        _kwargs["response_mime_type"] = kwargs.get("response_mime_type") or self.response_mime_type
        _kwargs["response_schema"] = kwargs.get("response_schema") or self.response_schema
        _kwargs["json_mode"] = kwargs.get("json_mode") or self.json_mode
        _kwargs["n_results"] = kwargs.get("n_results") or self.n_results
        _kwargs["tools"] = kwargs.get("tools") or self.tools
        _kwargs["tool_configs"] = kwargs.get("tool_configs") or self.tool_configs
        _kwargs["tool_choice"] = kwargs.get("tool_choice") or self.tool_choice

        return _kwargs

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        system_instruction: str | None = None,
        model_name: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_mime_type: str | None = None,
        json_mode: bool = False,
        response_schema: Mapping[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: str = "auto",
        tool_only: bool = False,
        n_results: int | None = None,
        **kwargs,
    ) -> CompletionResults | FunctionCallingResults:
        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            json_mode=json_mode,
            n_results=n_results,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            **kwargs,
        )

        if generation_kwargs.get("model_name") is None:
            raise ValueError("model_name must be provided")

        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        total_usage = Usage(model_name=generation_kwargs.get("model_name"))

        if generation_kwargs.get("tools"):
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt,
                init_conversation=init_conversation,
                **generation_kwargs,
            )

            if tool_only:
                self._clear_memory()

                return response_function_calling

            if response_function_calling.results:
                prompt = Message(
                    role="function",
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
            prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

        total_usage += response_chat.usage
        response_chat.usage = total_usage

        self._clear_memory()

        return response_chat

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        system_instruction: str | None = None,
        model_name: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_mime_type: str | None = None,
        json_mode: bool = False,
        response_schema: Mapping[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: str = "auto",
        tool_only: bool = False,
        n_results: int | None = None,
        **kwargs,
    ) -> CompletionResults | FunctionCallingResults:
        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            json_mode=json_mode,
            n_results=n_results,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            **kwargs,
        )

        if generation_kwargs.get("model_name") is None:
            raise ValueError("model_name must be provided")

        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        total_usage = Usage(model_name=generation_kwargs.get("model_name"))

        if generation_kwargs.get("tools"):
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt,
                init_conversation=init_conversation,
                **generation_kwargs,
            )

            if tool_only:
                self._clear_memory()

                return response_function_calling

            if response_function_calling.results:
                prompt = Message(
                    role="function",
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
            prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

        total_usage += response_chat.usage
        response_chat.usage = total_usage

        self._clear_memory()

        return response_chat

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        system_instruction: str | None = None,
        model_name: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_mime_type: str | None = None,
        json_mode: bool = False,
        response_schema: Mapping[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> Generator[CompletionResults, None, None]:
        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            json_mode=json_mode,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            **kwargs,
        )

        if generation_kwargs.get("model_name") is None:
            raise ValueError("model_name must be provided")

        total_usage = Usage(model_name=generation_kwargs.get("model_name"))

        if generation_kwargs.get("tools"):
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt,
                init_conversation=init_conversation,
                **generation_kwargs,
            )

            if response_function_calling.results:
                prompt = Message(
                    role="function",
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
            prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

        for result in response_chat:
            if result.usage.total_tokens > 0:
                total_usage += result.usage
                result.usage = total_usage
            yield result

        self._clear_memory()

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        system_instruction: str | None = None,
        model_name: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_mime_type: str | None = None,
        json_mode: bool = False,
        response_schema: Mapping[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> AsyncGenerator[CompletionResults, None]:
        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            json_mode=json_mode,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            **kwargs,
        )

        if generation_kwargs.get("model_name") is None:
            raise ValueError("model_name must be provided")

        total_usage = Usage(model_name=generation_kwargs.get("model_name"))

        if generation_kwargs.get("tools"):
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt,
                init_conversation=init_conversation,
                **generation_kwargs,
            )

            if response_function_calling.results:
                prompt = Message(
                    role="function",
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
            prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

        async for result in response_chat:
            if result.usage.total_tokens > 0:
                total_usage += result.usage
                result.usage = total_usage
            yield result

        self._clear_memory()
