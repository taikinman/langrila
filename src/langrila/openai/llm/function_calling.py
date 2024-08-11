import copy
import json
from typing import Callable

from openai._types import NOT_GIVEN, NotGiven

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
from ..model_config import (
    _OLDER_MODEL_CONFIG,
    MODEL_CONFIG,
    MODEL_POINT,
)
from ..openai_utils import get_async_client, get_client, get_token_limit
from ..tools import OpenAIToolConfig, ToolConfig


class FunctionCallingCoreModule(BaseFunctionCallingModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        api_type: str = "openai",
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
    ) -> None:
        assert api_type in ["openai", "azure"], "api_type must be 'openai' or 'azure'."
        if api_type == "azure":
            assert (
                api_version and endpoint_env_name and deployment_id_env_name
            ), "api_version, endpoint_env_name, and deployment_id_env_name must be specified for Azure API."

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
        if model_name not in _OLDER_MODEL_CONFIG.keys():
            self.seed = seed
            self.additional_inputs["seed"] = seed
            self.tool_configs = [f.format() for f in client_tool_config]
            self.additional_inputs["tools"] = self.tool_configs
        else:
            if seed:
                print(
                    f"seed is ignored because it's not supported for {model_name} (api_type:{api_type})"
                )
            self.tool_configs = [f.format()["function"] for f in client_tool_config]
            self.additional_inputs["functions"] = self.tool_configs

        if system_instruction:
            system_instruction = OpenAIMessage.to_universal_message(
                role="system", message=system_instruction
            )
            self.system_instruction = OpenAIMessage.to_client_message(system_instruction)
        else:
            self.system_instruction = None

        self.conversation_length_adjuster = conversation_length_adjuster

    def _get_client_tool_config_type(self) -> ToolConfig:
        return OpenAIToolConfig

    def _set_tool_choice(self, tool_choice: str = "auto"):
        if self.model_name not in _OLDER_MODEL_CONFIG.keys():
            self.additional_inputs["tool_choice"] = (
                str(tool_choice).lower()
                if tool_choice in ["auto", "required", None]
                else {"type": "function", "function": {"name": tool_choice}}
            )
        else:
            self.additional_inputs["function_call"] = tool_choice

    def run(
        self, messages: list[dict[str, str]], tool_choice: str = "auto"
    ) -> FunctionCallingResults:
        self._set_tool_choice(tool_choice)

        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        client = get_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        response = client.chat.completions.create(
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

        if self.model_name not in _OLDER_MODEL_CONFIG.keys():
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

        elif self.model_name in _OLDER_MODEL_CONFIG.keys():
            response_message = response.choices[0].message
            function_call = response_message.function_call

            output = []
            if function_call is not None:
                funcname = function_call.name
                args = function_call.arguments
                func_out = self.tools[funcname](**json.loads(args))

                output += [
                    ToolOutput(
                        call_id=None,
                        funcname=funcname,
                        args=args,
                        output=func_out,
                    )
                ]

            return FunctionCallingResults(
                usage=usage, results=output, prompt=copy.deepcopy(_messages)
            )

        else:
            raise ValueError(f"model_name {self.model_name} is not supported.")

    async def arun(
        self, messages: list[dict[str, str]], tool_choice: str = "auto"
    ) -> FunctionCallingResults:
        self._set_tool_choice(tool_choice)

        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        client = get_async_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        response = await client.chat.completions.create(
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

        if self.model_name not in _OLDER_MODEL_CONFIG.keys():
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

        elif self.model_name in _OLDER_MODEL_CONFIG.keys():
            response_message = response.choices[0].message
            function_call = response_message.function_call

            output = []
            if function_call is not None:
                funcname = function_call.name
                args = function_call.arguments
                func_out = self.tools[funcname](**json.loads(args))

                output += [
                    ToolOutput(
                        call_id=None,
                        funcname=funcname,
                        args=args,
                        output=func_out,
                    )
                ]

            return FunctionCallingResults(
                usage=usage, results=output, prompt=copy.deepcopy(_messages)
            )

        else:
            raise ValueError(f"model_name {self.model_name} is not supported.")


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
        context_length: int | None = None,
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
        if model_name in MODEL_POINT.keys():
            print(f"{model_name} is automatically converted to {MODEL_POINT[model_name]}")
            model_name = MODEL_POINT[model_name]

        assert (
            model_name in MODEL_CONFIG.keys()
        ), f"model_name must be one of {', '.join(sorted(MODEL_CONFIG.keys()))}."

        token_lim = get_token_limit(model_name)
        max_tokens = max_tokens if max_tokens else MODEL_CONFIG[model_name]["max_output_tokens"]
        context_length = token_lim - max_tokens if context_length is None else context_length
        assert (
            token_lim >= max_tokens + context_length
        ), f"max_tokens({max_tokens}) + context_length({context_length}) must be less than or equal to the token limit of the model ({token_lim})."
        assert context_length > 0, "context_length must be positive."

        conversation_length_adjuster = (
            OldConversationTruncationModule(model_name=model_name, context_length=context_length)
            if conversation_length_adjuster is None
            else conversation_length_adjuster
        )

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
