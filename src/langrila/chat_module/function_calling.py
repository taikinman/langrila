import asyncio
import json
from typing import Callable, Optional

from pydantic import BaseModel, field_validator

from ..base import BaseConversationLengthAdjuster, BaseFilter, BaseModule
from ..conversation_adjuster.truncate import OldConversationTruncationModule
from ..message import Message
from ..model_config import _NEWER_MODEL_CONFIG, _OLDER_MODEL_CONFIG, MODEL_CONFIG, MODEL_POINT
from ..result import FunctionCallingResults, ToolOutput
from ..usage import Usage
from ..utils import get_async_client, get_client, get_token_limit, make_batch


class ToolProperty(BaseModel):
    name: str
    type: str
    description: str

    def model_dump(self):
        return {self.name: super().model_dump(exclude=["name"])}

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


class BaseFunctionCallingModule(BaseModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        tool_choice: str = "auto",
        api_type: str = "openai",
        api_version: Optional[str] = None,
        endpoint_env_name: Optional[str] = None,
        deployment_id_env_name: Optional[str] = None,
        organization_id_env_name: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 2,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
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

        self.tool_choice = tool_choice
        self.max_tokens = max_tokens

        self.additional_inputs = {}
        if model_name in _NEWER_MODEL_CONFIG.keys():
            self.seed = seed
            self.additional_inputs["seed"] = seed
            self.tool_configs = [f.model_dump() for f in tool_configs]
            self.additional_inputs["tools"] = self.tool_configs
            self.additional_inputs["tool_choice"] = self.tool_choice
        else:
            if seed:
                print(
                    f"seed is ignored because it's not supported for {model_name} (api_type:{api_type})"
                )
            self.tool_configs = [f.model_dump()["function"] for f in tool_configs]
            self.additional_inputs["functions"] = self.tool_configs
            self.additional_inputs["function_call"] = self.tool_choice

    def run(self, messages: list[dict[str, str]]) -> FunctionCallingResults:
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

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages if isinstance(messages, list) else [messages],
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            **self.additional_inputs,
        )

        usage = Usage()
        usage += response.usage

        if self.model_name in _NEWER_MODEL_CONFIG.keys():
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            results = []
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

            return FunctionCallingResults(usage=usage, results=results, prompt=messages)
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

            return FunctionCallingResults(usage=usage, results=output, prompt=messages)

    async def arun(self, messages: list[dict[str, str]]) -> FunctionCallingResults:
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

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages if isinstance(messages, list) else [messages],
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            **self.additional_inputs,
        )

        usage = Usage()
        usage += response.usage

        if self.model_name in _NEWER_MODEL_CONFIG.keys():
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            results = []
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

            return FunctionCallingResults(usage=usage, results=results, prompt=messages)
        elif self.model_name in _OLDER_MODEL_CONFIG.keys():
            response_message = response.choices[0].message
            funcname = response_message.function_call.name
            args = response_message.function_call.arguments
            func_out = self.tools[funcname](**json.loads(args))

            output = ToolOutput(
                call_id=None,
                funcname=funcname,
                args=args,
                output=func_out,
            )

            return FunctionCallingResults(usage=usage, results=[output], prompt=messages)


class OpenAIFunctionCallingModule(BaseModule):
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
        content_filter: Optional[BaseFilter] = None,
        conversation_length_adjuster: Optional[BaseConversationLengthAdjuster] = None,
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

        self.function_calling_model = BaseFunctionCallingModule(
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
        )

        self.conversation_length_adjuster = (
            OldConversationTruncationModule(model_name=model_name, context_length=context_length)
            if conversation_length_adjuster is None
            else conversation_length_adjuster
        )
        self.content_filter = content_filter

    def run(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
    ) -> FunctionCallingResults:
        messages: list[dict[str, str]] = []

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt).as_user)

        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)

        messages = self.conversation_length_adjuster(messages)

        response = self.function_calling_model(messages)

        if self.content_filter is not None:
            for i, r in enumerate(response.results):
                response.results[i].args = self.content_filter.restore([response.results[i].args])[
                    0
                ]

        return response

    async def arun(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
    ) -> FunctionCallingResults:
        messages: list[dict[str, str]] = []

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt).as_user)

        if self.content_filter is not None:
            messages = await self.content_filter(messages, arun=True)

        messages = self.conversation_length_adjuster(messages)

        response = await self.function_calling_model(messages, arun=True)

        if self.content_filter is not None:
            for i, r in enumerate(response.results):
                response.results[i].args = self.content_filter.restore([response.results[i].args])[
                    0
                ]

        return response

    async def abatch_run(
        self,
        prompts: list[str],
        init_conversations: Optional[list[list[dict[str, str]]]] = None,
        batch_size: int = 4,
    ) -> list[FunctionCallingResults]:
        if init_conversations is None:
            init_conversations = [None] * len(prompts)

        z = zip(prompts, init_conversations, strict=True)
        batches = make_batch(list(z), batch_size)
        results = []
        for batch in batches:
            async_processes = [
                self.arun(prompt=prompt, init_conversation=init_conversation)
                for prompt, init_conversation in batch
            ]
            results.extend(await asyncio.gather(*async_processes))
        return results
