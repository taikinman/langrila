import copy
import json
import os
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from google.auth import credentials as auth_credentials
from pydantic import BaseModel

from ...base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseFunctionCallingModule,
    BaseMessage,
)
from ...llm_wrapper import FunctionCallingWrapperModule
from ...message_content import ConversationType, InputType
from ...result import FunctionCallingResults, ToolCallResponse, ToolOutput
from ...tools import ToolConfig
from ...usage import TokenCounter, Usage
from ...utils import generate_dummy_call_id
from ..gemini_utils import (
    get_call_config,
    get_client,
    get_client_tool_type,
    get_message_cls,
    get_tool_cls,
)


class GeminiFunctionCallingCoreModule(BaseFunctionCallingModule):
    def __init__(
        self,
        api_key_env_name: str | None = None,
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
        **kwargs: Any,
    ):
        self.api_type = api_type
        self._client = get_client(
            api_key_env_name=api_key_env_name,
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
        )

    def run(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> FunctionCallingResults:
        runnable_tools_dict = kwargs.pop("runnable_tools_dict")
        response = self._client.generate_message(
            contents=messages,
            **kwargs,
        )

        parts = response.candidates[0].content.parts

        results = []
        calls = []
        for part in parts:
            if fn := part.function_call:
                funcname = fn.name
                args = dict(fn.args)
                func_out = runnable_tools_dict[funcname](**args)
                dummy_call_id = generate_dummy_call_id(24)
                output = ToolOutput(
                    call_id=dummy_call_id,
                    funcname=funcname,
                    args=json.dumps(args),
                    output=func_out,
                )

                call = ToolCallResponse(
                    name=funcname,
                    args=args,
                    call_id=dummy_call_id,
                )
                results.append(output)
                calls.append(call)

        usage_metadata = response.usage_metadata

        return FunctionCallingResults(
            usage=Usage(
                model_name=kwargs.get("model_name"),
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            results=results,
            calls=calls,
            prompt=copy.deepcopy(messages),
            raw=response,
        )

    async def arun(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> FunctionCallingResults:
        runnable_tools_dict = kwargs.pop("runnable_tools_dict")

        response = await self._client.generate_message_async(
            contents=messages,
            **kwargs,
        )

        parts = response.candidates[0].content.parts

        results = []
        calls = []
        for part in parts:
            if fn := part.function_call:
                funcname = fn.name
                args = dict(fn.args)
                func_out = runnable_tools_dict[funcname](**args)
                dummy_call_id = generate_dummy_call_id(24)
                output = ToolOutput(
                    call_id=dummy_call_id,
                    funcname=funcname,
                    args=json.dumps(args),
                    output=func_out,
                )

                call = ToolCallResponse(
                    name=funcname,
                    args=args,
                    call_id=dummy_call_id,
                )
                results.append(output)
                calls.append(call)

        usage_metadata = response.usage_metadata

        return FunctionCallingResults(
            usage=Usage(
                model_name=kwargs.get("model_name"),
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            results=results,
            calls=calls,
            prompt=copy.deepcopy(messages),
            raw=response,
        )


class GeminiFunctionCallingModule(FunctionCallingWrapperModule):
    def __init__(
        self,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        api_key_env_name: str | None = None,
        max_output_tokens: int | None = None,
        timeout: int | None = None,
        content_filter: Optional[BaseFilter] = None,
        conversation_memory: Optional[BaseConversationMemory] = None,
        token_counter: Optional[TokenCounter] = None,
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
        system_instruction: str | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        seed: int | None = None,
        stop_sequences: Iterable[str] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.tools = tools
        self.tool_configs = tool_configs
        self.api_key_env_name = api_key_env_name
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
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
        self.system_instruction = system_instruction
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.routing_config = routing_config
        self.logprobs = logprobs
        self.response_logprobs = response_logprobs
        self.seed = seed
        self.stop_sequences = stop_sequences
        self.tool_choice = tool_choice

        # The module to call client API
        function_calling_model = GeminiFunctionCallingCoreModule(
            api_key_env_name=api_key_env_name,
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
            **kwargs,
        )

        super().__init__(
            function_calling_model=function_calling_model,
            content_filter=content_filter,
            conversation_memory=conversation_memory,
            token_counter=token_counter,
        )

    def _get_call_config(self, tool_choice: str | None = "auto"):
        return get_call_config(api_type=self.api_type, tool_choice=tool_choice)

    def _get_client_tool_config_type(self, api_type: str):
        return get_client_tool_type(api_type=api_type)

    def _get_generation_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        _kwargs = {}
        _kwargs["system_instruction"] = kwargs.get("system_instruction") or self.system_instruction
        _kwargs["model_name"] = kwargs.get("model_name") or self.model_name
        _kwargs["temperature"] = kwargs.get("temperature") or self.temperature
        _kwargs["top_p"] = kwargs.get("top_p") or self.top_p
        _kwargs["top_k"] = kwargs.get("top_k") or self.top_k
        _kwargs["stop_sequences"] = kwargs.get("stop_sequences") or self.stop_sequences
        _kwargs["frequency_penalty"] = kwargs.get("frequency_penalty ") or self.frequency_penalty
        _kwargs["presence_penalty"] = kwargs.get("presence_penalty") or self.presence_penalty
        _kwargs["seed"] = kwargs.get("seed") or self.seed
        _kwargs["max_output_tokens"] = kwargs.get("max_output_tokens") or self.max_output_tokens
        _kwargs["routing_config"] = kwargs.get("routing_config") or self.routing_config
        _kwargs["logprobs"] = kwargs.get("logprobs") or self.logprobs
        _kwargs["response_logprobs"] = kwargs.get("response_logprobs") or self.response_logprobs
        _kwargs["candidate_count"] = 1
        _kwargs["response_mime_type"] = "text/plain"

        if self.api_type == "genai":
            from google.generativeai.types.helper_types import RequestOptions

            _kwargs["request_options"] = RequestOptions(
                timeout=kwargs.get("timeout") or 60,
            )

        _kwargs["tool_config"] = self._get_call_config(
            tool_choice=kwargs.get("tool_choice") or self.tool_choice
        )

        _tools = kwargs.get("tools") or self.tools
        _tool_configs = kwargs.get("tool_configs") or self.tool_configs

        if not (_tool_configs and _tools):
            raise ValueError("tool_configs must be provided.")

        ClientToolConfig = self._get_client_tool_config_type(self.api_type)
        client_tool_configs = ClientToolConfig.from_universal_configs(_tool_configs)
        tool_cls = get_tool_cls(api_type=self.api_type)
        function_declarations = [config.format() for config in client_tool_configs]
        tool_configs_declaration = [tool_cls(function_declarations=function_declarations)]
        runnable_tools_dict = self._set_runnable_tools_dict(_tools)

        _kwargs["tools"] = tool_configs_declaration
        _kwargs["runnable_tools_dict"] = runnable_tools_dict

        return _kwargs

    def _get_client_message_type(self) -> type[BaseMessage]:
        return get_message_cls(self.api_type)

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
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> FunctionCallingResults:
        generation_kwargs = self._get_generation_kwargs(
            prompt=prompt,
            init_conversation=init_conversation,
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
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
        system_instruction: str | None = None,
        model_name: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> FunctionCallingResults:
        generation_kwargs = self._get_generation_kwargs(
            prompt=prompt,
            init_conversation=init_conversation,
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            **kwargs,
        )
        return await super().arun(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )
