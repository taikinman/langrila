import copy
import json
from typing import Any, Callable, Literal, Mapping

import httpx
from openai._types import NOT_GIVEN, NotGiven
from openai.lib.azure import AzureADTokenProvider

from ...base import (
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseFunctionCallingModule,
)
from ...llm_wrapper import FunctionCallingWrapperModule
from ...message_content import ConversationType, InputType
from ...result import FunctionCallingResults, ToolCallResponse, ToolOutput
from ...usage import TokenCounter, Usage
from ...warnings import deprecated_argument
from ..conversation_adjuster.truncate import OldConversationTruncationModule
from ..message import OpenAIMessage
from ..openai_utils import get_client
from ..tools import OpenAIToolConfig, ToolConfig


class FunctionCallingCoreModule(BaseFunctionCallingModule):
    def __init__(
        self,
        api_key_env_name: str,
        api_type: Literal["openai", "azure"] = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        organization_id_env_name: str | None = None,
        timeout: int = 30,
        max_retries: int = 2,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AzureADTokenProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
        **kwargs: Any,
    ) -> None:
        self.conversation_length_adjuster = conversation_length_adjuster

        self._client = get_client(
            api_key_env_name=api_key_env_name,
            organization_id_env_name=organization_id_env_name,
            api_version=api_version,
            endpoint_env_name=endpoint_env_name,
            deployment_id_env_name=deployment_id_env_name,
            api_type=api_type,
            max_retries=max_retries,
            timeout=timeout,
            project=project,
            base_url=base_url,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )

    def _get_client_tool_config_type(self) -> ToolConfig:
        return OpenAIToolConfig

    def run(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> FunctionCallingResults:
        runnable_tools_dict = kwargs.pop("runnable_tools_dict")

        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        _messages = (
            [kwargs.get("system_instruction")] + messages
            if kwargs.get("system_instruction")
            else messages
        )

        _conversation_length_adjuster = (
            kwargs.pop("conversation_length_adjuster", None) or self.conversation_length_adjuster
        )
        if _conversation_length_adjuster:
            _messages = _conversation_length_adjuster.run(_messages)

        response = self._client.generate_message(
            messages=_messages,
            stream=False,
            **kwargs,
        )

        usage = Usage(model_name=kwargs.get("model"))
        usage += response.usage

        response_message = response.choices[0].message
        self._response_message = response_message
        tool_calls = response_message.tool_calls

        results = []
        calls = []
        if tool_calls is not None:
            for tool_call in tool_calls:
                call_id = tool_call.id
                funcname = tool_call.function.name
                args = tool_call.function.arguments
                func_out = runnable_tools_dict[funcname](**json.loads(args))
                output = ToolOutput(
                    call_id=call_id,
                    funcname=funcname,
                    args=args,
                    output=func_out,
                )

                results.append(output)

                call = ToolCallResponse(
                    call_id=call_id,
                    name=funcname,
                    args=args,
                )

                calls.append(call)

        return FunctionCallingResults(
            usage=usage, results=results, prompt=copy.deepcopy(_messages), calls=calls, raw=response
        )

    async def arun(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> FunctionCallingResults:
        runnable_tools_dict = kwargs.pop("runnable_tools_dict")

        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        _messages = (
            [kwargs.get("system_instruction")] + messages
            if kwargs.get("system_instruction")
            else messages
        )

        _conversation_length_adjuster = (
            kwargs.pop("conversation_length_adjuster", None) or self.conversation_length_adjuster
        )
        if _conversation_length_adjuster:
            _messages = _conversation_length_adjuster.run(_messages)

        response = await self._client.generate_message_async(
            messages=_messages,
            stream=False,
            **kwargs,
        )

        usage = Usage(model_name=kwargs.get("model"))
        usage += response.usage

        response_message = response.choices[0].message
        self._response_message = response_message
        tool_calls = response_message.tool_calls

        results = []
        calls = []
        if tool_calls is not None:
            for tool_call in tool_calls:
                call_id = tool_call.id
                funcname = tool_call.function.name
                args = tool_call.function.arguments
                func_out = runnable_tools_dict[funcname](**json.loads(args))
                output = ToolOutput(
                    call_id=call_id,
                    funcname=funcname,
                    args=args,
                    output=func_out,
                )

                results.append(output)

                call = ToolCallResponse(
                    call_id=call_id,
                    name=funcname,
                    args=args,
                )

                calls.append(call)

        return FunctionCallingResults(
            usage=usage, results=results, prompt=copy.deepcopy(_messages), calls=calls, raw=response
        )


class OpenAIFunctionCallingModule(FunctionCallingWrapperModule):
    @deprecated_argument(
        arg="context_length",
        removal="1.0.0",
        since="0.4.0",
        alternative="conversation_length_adjuster",
        module_name="OpenAIFunctionCallingModule",
        details="Token management section in langrila/notebooks/01.introduction.ipynb",
    )
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        tool_choice: str | Literal["auto", "none"] = "auto",
        organization_id_env_name: str | None = None,
        api_type: str = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        timeout: int = 60,
        max_retries: int = 2,
        seed: int | NotGiven = NOT_GIVEN,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        system_instruction: str | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        context_length: int | None = None,
        token_counter: TokenCounter | None = None,
        top_p: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AzureADTokenProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
        **kwargs: Any,
    ) -> None:
        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.tools = tools
        self.tool_configs = tool_configs
        self.parallel_tool_calls = parallel_tool_calls
        self.tool_choice = tool_choice
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
        self.system_instruction = system_instruction
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.user = user
        self.stop = stop

        if conversation_length_adjuster is None and context_length:
            if model_name is None:
                raise ValueError("model_name must be specified if context_length is specified.")

            self.conversation_length_adjuster = OldConversationTruncationModule(
                context_length=context_length, model_name=model_name
            )
        else:
            self.conversation_length_adjuster = conversation_length_adjuster

        # The module to call client API
        function_calling_model = FunctionCallingCoreModule(
            api_key_env_name=api_key_env_name,
            organization_id_env_name=organization_id_env_name,
            api_type=api_type,
            api_version=api_version,
            endpoint_env_name=endpoint_env_name,
            deployment_id_env_name=deployment_id_env_name,
            timeout=timeout,
            max_retries=max_retries,
            conversation_length_adjuster=conversation_length_adjuster,
            project=project,
            base_url=base_url,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
            **kwargs,
        )

        super().__init__(
            function_calling_model=function_calling_model,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
        )

    def _system_instruction_to_message(self, system_instruction: str | None) -> OpenAIMessage:
        if system_instruction:
            _system_instruction = OpenAIMessage.to_universal_message(
                role="system", message=system_instruction
            )
            return OpenAIMessage.to_client_message(_system_instruction)
        else:
            return None

    def _get_client_message_type(self) -> OpenAIMessage:
        return OpenAIMessage

    def _get_client_tool_config_type(self) -> ToolConfig:
        return OpenAIToolConfig

    def _get_tool_choice_dict(self, tool_choice: str | Literal["auto", "none"]) -> dict[str, Any]:
        if tool_choice in ["auto", "none"]:
            return tool_choice
        else:
            return {"type": "function", "function": {"name": tool_choice}}

    def _get_generation_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        _kwargs = {}
        _kwargs["system_instruction"] = self._system_instruction_to_message(
            kwargs.get("system_instruction") or self.system_instruction
        )
        _kwargs["model"] = kwargs.get("model_name") or self.model_name
        _kwargs["temperature"] = kwargs.get("temperature") or self.temperature
        _kwargs["top_p"] = kwargs.get("top_p") or self.top_p
        _kwargs["stop"] = kwargs.get("stop") or self.stop
        _kwargs["frequency_penalty"] = kwargs.get("frequency_penalty ") or self.frequency_penalty
        _kwargs["presence_penalty"] = kwargs.get("presence_penalty") or self.presence_penalty
        _kwargs["user"] = kwargs.get("user") or self.user
        _kwargs["seed"] = kwargs.get("seed") or self.seed
        _kwargs["n"] = 1
        _kwargs["parallel_tool_calls"] = (
            kwargs.get("parallel_tool_calls") or self.parallel_tool_calls
        )

        _kwargs["max_tokens"] = kwargs.get("max_tokens") or self.max_tokens
        _kwargs["max_completion_tokens"] = (
            kwargs.get("max_completion_tokens") or self.max_completion_tokens
        )

        _kwargs["tool_choice"] = self._get_tool_choice_dict(
            kwargs.get("tool_choice") or self.tool_choice
        )

        _tools = kwargs.get("tools") or self.tools
        _tool_configs = kwargs.get("tool_configs") or self.tool_configs

        if not (_tool_configs and _tools):
            raise ValueError("tool_configs must be provided.")

        _tools_dict = self._set_runnable_tools_dict(_tools)
        ClientToolConfig = self._get_client_tool_config_type()
        client_tool_config = ClientToolConfig.from_universal_configs(_tool_configs)
        _tool_names_from_config = {f.name for f in _tool_configs}
        _kwargs["tools"] = [f.format() for f in client_tool_config]
        _kwargs["runnable_tools_dict"] = _tools_dict

        assert (
            len(_tool_names_from_config ^ set(_tools_dict.keys())) == 0
        ), f"tool names in tool_configs must be the same as the function names \
            in tools. tool names in tool_configs: {_tool_names_from_config}, \
            function names in tools: {set(_tools_dict.keys())}"

        return _kwargs

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
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
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: str | Literal["auto", "none"] = "auto",
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> FunctionCallingResults:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            system_instruction=system_instruction,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            seed=seed,
            **kwargs,
        )

        return super().run(
            prompt=prompt,
            init_conversation=init_conversation,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            **generation_kwargs,
        )

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        model_name: str | None = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        stop: str | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        system_instruction: str | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: str | Literal["auto", "none"] = "auto",
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> FunctionCallingResults:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            system_instruction=system_instruction,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            seed=seed,
            **kwargs,
        )

        return await super().arun(
            prompt=prompt,
            init_conversation=init_conversation,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            **generation_kwargs,
        )
