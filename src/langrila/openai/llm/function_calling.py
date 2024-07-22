import asyncio
import copy
import json
from typing import Callable, Optional

from pydantic import BaseModel, field_validator

from ...base import (
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseFunctionCallingModule,
    BaseMessage,
)
from ...llm_wrapper import FunctionCallingWrapperModule
from ...mixin import ConversationMixin, FilterMixin
from ...result import FunctionCallingResults, ToolCallResponse, ToolOutput
from ...usage import TokenCounter, Usage
from ...utils import make_batch
from ..conversation_adjuster.truncate import OldConversationTruncationModule
from ..message import OpenAIMessage
from ..model_config import (
    _NEWER_MODEL_CONFIG,
    _OLDER_MODEL_CONFIG,
    MODEL_CONFIG,
    MODEL_POINT,
)
from ..openai_utils import get_async_client, get_client, get_token_limit


class ToolProperty(BaseModel):
    name: str
    type: str
    description: str
    enum: list[str | int | float] | None = None

    def model_dump(self):
        return {
            self.name: super().model_dump(exclude=["name"]) | {"enum": self.enum}
            if self.enum
            else {}
        }

    @field_validator("type")
    def check_type_value(cls, v):
        if v not in {"string", "number", "boolean"}:
            raise ValueError("type must be one of string or number.")

        return v


class ToolParameter(BaseModel):
    type: str = "object"
    properties: list[ToolProperty]
    required: Optional[list[str]] = None

    def model_dump(self):
        dumped = super().model_dump(exclude=["properties", "required"])

        _properties = {}
        for p in self.properties:
            _properties.update(p.model_dump())
        dumped["properties"] = _properties

        if self.required is not None:
            dumped["required"] = self.required
        return dumped

    @field_validator("type")
    def check_type_value(cls, v):
        if v not in {"object"}:
            raise ValueError("supported type is only object")

        return v

    @field_validator("required")
    def check_required_value(cls, required, values):
        properties = values.data["properties"]
        property_names = {p.name for p in properties}
        if required is not None:
            for r in required:
                if r not in property_names:
                    raise ValueError(f"required property '{r}' is not defined in properties.")
        return required


class ToolConfig(BaseModel):
    name: str
    type: str = "function"
    description: str
    parameters: ToolParameter

    def model_dump(self):
        dumped = super().model_dump(exclude=["parameters", "type"])
        dumped["parameters"] = self.parameters.model_dump()
        return {"type": self.type, self.type: dumped}

    @field_validator("type")
    def check_type_value(cls, v):
        if v not in {"function"}:
            raise ValueError("supported type is only function")

        return v


class FunctionCallingCoreModule(BaseFunctionCallingModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        api_type: str = "openai",
        api_version: Optional[str] = None,
        endpoint_env_name: Optional[str] = None,
        deployment_id_env_name: Optional[str] = None,
        organization_id_env_name: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 2,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
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
        self.tools = {f.__name__: f for f in tools}

        _tool_names_from_config = {f.name for f in tool_configs}
        assert (
            len(_tool_names_from_config ^ set(self.tools.keys())) == 0
        ), f"tool names in tool_configs must be the same as the function names in tools. tool names in tool_configs: {_tool_names_from_config}, function names in tools: {set(self.tools.keys())}"

        self.max_tokens = max_tokens

        self.additional_inputs = {}
        if model_name not in _OLDER_MODEL_CONFIG.keys():
            self.seed = seed
            self.additional_inputs["seed"] = seed
            self.tool_configs = [f.model_dump() for f in tool_configs]
            self.additional_inputs["tools"] = self.tool_configs
        else:
            if seed:
                print(
                    f"seed is ignored because it's not supported for {model_name} (api_type:{api_type})"
                )
            self.tool_configs = [f.model_dump()["function"] for f in tool_configs]
            self.additional_inputs["functions"] = self.tool_configs

        self.system_instruction = (
            OpenAIMessage(content=system_instruction).as_system if system_instruction else None
        )
        self.conversation_length_adjuster = conversation_length_adjuster

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
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
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
                usage=usage, results=results, prompt=copy.deepcopy(messages), calls=calls
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
                usage=usage, results=output, prompt=copy.deepcopy(messages)
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
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
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
                usage=usage, results=results, prompt=copy.deepcopy(messages), calls=calls
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
                usage=usage, results=output, prompt=copy.deepcopy(messages)
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
        organization_id_env_name: Optional[str] = None,
        api_type: str = "openai",
        api_version: Optional[str] = None,
        endpoint_env_name: Optional[str] = None,
        deployment_id_env_name: Optional[str] = None,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        seed: Optional[int] = None,
        context_length: Optional[int] = None,
        conversation_memory: Optional[BaseConversationMemory] = None,
        content_filter: Optional[BaseFilter] = None,
        system_instruction: Optional[str] = None,
        conversation_length_adjuster: Optional[BaseConversationLengthAdjuster] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        if model_name in MODEL_POINT.keys():
            print(f"{model_name} is automatically converted to {MODEL_POINT[model_name]}")
            model_name = MODEL_POINT[model_name]

        assert (
            model_name in MODEL_CONFIG.keys()
        ), f"model_name must be one of {', '.join(sorted(MODEL_CONFIG.keys()))}."

        token_lim = get_token_limit(model_name)
        max_tokens = max_tokens if max_tokens else int(token_lim / 2)
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
        )

        content_filter = content_filter
        conversation_memory = conversation_memory

        super().__init__(
            function_calling_model=function_calling_model,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
        )

    def _get_client_message_type(self) -> type[BaseMessage]:
        return OpenAIMessage
