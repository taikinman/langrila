import copy
import json
from typing import Callable, Literal, Mapping

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
from ...result import FunctionCallingResults, ToolCallResponse, ToolOutput
from ...usage import TokenCounter, Usage
from ..conversation_adjuster.truncate import OldConversationTruncationModule
from ..message import OpenAIMessage
from ..openai_utils import get_client, get_token_limit
from ..tools import OpenAIToolConfig, ToolConfig


class FunctionCallingCoreModule(BaseFunctionCallingModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        api_type: Literal["openai", "azure"] = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        organization_id_env_name: str | None = None,
        timeout: int = 30,
        max_retries: int = 2,
        max_tokens: int = 2048,
        seed: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        system_instruction: str | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
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
        self.organization_id_env_name = organization_id_env_name
        self.api_type = api_type
        self.api_version = api_version
        self.endpoint_env_name = endpoint_env_name
        self.deployment_id_env_name = deployment_id_env_name
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.user = user

        self.tools = self._set_runnable_tools_dict(tools)

        _tool_names_from_config = {f.name for f in tool_configs}
        assert (
            len(_tool_names_from_config ^ set(self.tools.keys())) == 0
        ), f"tool names in tool_configs must be the same as the function names in tools. tool names in tool_configs: {_tool_names_from_config}, function names in tools: {set(self.tools.keys())}"

        self.max_tokens = max_tokens

        ClientToolConfig = self._get_client_tool_config_type()
        client_tool_config = ClientToolConfig.from_universal_configs(tool_configs)

        self.additional_inputs = {}
        self.seed = seed
        self.additional_inputs["seed"] = seed
        self.tool_configs = [f.format() for f in client_tool_config]
        self.additional_inputs["tools"] = self.tool_configs

        if system_instruction:
            system_instruction = OpenAIMessage.to_universal_message(
                role="system", message=system_instruction
            )
            self.system_instruction = OpenAIMessage.to_client_message(system_instruction)
        else:
            self.system_instruction = None

        self.conversation_length_adjuster = conversation_length_adjuster

        self._client = get_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
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

    def _set_tool_choice(self, tool_choice: str = "auto"):
        self.additional_inputs["tool_choice"] = (
            str(tool_choice).lower()
            if tool_choice in ["auto", "required", None]
            else {"type": "function", "function": {"name": tool_choice}}
        )

    def run(
        self, messages: list[dict[str, str]], tool_choice: str = "auto"
    ) -> FunctionCallingResults:
        self._set_tool_choice(tool_choice)

        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        response = self._client.generate_message(
            model=self.model_name,
            messages=_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=NOT_GIVEN,
            stream=False,
            user=self.user,
            **self.additional_inputs,
        )

        usage = Usage(model_name=self.model_name)
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
                func_out = self.tools[funcname](**json.loads(args))
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
            usage=usage, results=results, prompt=copy.deepcopy(_messages), calls=calls
        )

    async def arun(
        self, messages: list[dict[str, str]], tool_choice: str = "auto"
    ) -> FunctionCallingResults:
        self._set_tool_choice(tool_choice)

        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        response = await self._client.generate_message_async(
            model=self.model_name,
            messages=_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=NOT_GIVEN,
            stream=False,
            user=self.user,
            **self.additional_inputs,
        )

        usage = Usage(model_name=self.model_name)
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
                func_out = self.tools[funcname](**json.loads(args))
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
            usage=usage, results=results, prompt=copy.deepcopy(_messages), calls=calls
        )


class OpenAIFunctionCallingModule(FunctionCallingWrapperModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        organization_id_env_name: str | None = None,
        api_type: str = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        seed: int | NotGiven = NOT_GIVEN,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        system_instruction: str | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        token_counter: TokenCounter | None = None,
        top_p: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
    ) -> None:
        # The module to call client API
        function_calling_model = FunctionCallingCoreModule(
            api_key_env_name=api_key_env_name,
            organization_id_env_name=organization_id_env_name,
            api_type=api_type,
            api_version=api_version,
            endpoint_env_name=endpoint_env_name,
            deployment_id_env_name=deployment_id_env_name,
            tools=tools,
            tool_configs=tool_configs,
            model_name=model_name,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
            system_instruction=system_instruction,
            conversation_length_adjuster=conversation_length_adjuster,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
        )

        content_filter = content_filter
        conversation_memory = conversation_memory

        super().__init__(
            function_calling_model=function_calling_model,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
        )

    def _get_client_message_type(self) -> OpenAIMessage:
        return OpenAIMessage
