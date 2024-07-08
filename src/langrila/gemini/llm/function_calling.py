import json
from typing import AsyncGenerator, Callable, Generator, Optional

import google.ai.generativelanguage as glm
import google.generativeai as genai
from google.generativeai.types.generation_types import GenerationConfig
from google.generativeai.types.helper_types import RequestOptions
from pydantic import BaseModel

from ...base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseMessage,
)
from ...llm_wrapper import FunctionCallingWrapperModule
from ...result import FunctionCallingResults, ToolOutput
from ...usage import TokenCounter, Usage
from ..gemini_utils import get_model
from ..message import GeminiMessage


class ToolProperty(BaseModel):
    name: str
    type: str
    description: str

    def model_dump(self):
        return {
            self.name: {"type_": self.type.upper(), "description": self.description},
        }


class ToolParameter(BaseModel):
    type: str = "object"
    properties: list[ToolProperty]
    required: Optional[list[str]] = []

    def model_dump(self):
        property_descriptions = {}
        property_types = {}
        for prop in self.properties:
            prop_dict = prop.model_dump()
            for key, value in prop_dict.items():
                property_descriptions.update({key: value.pop("description")})
                property_types.update({key: {"type_": value.pop("type_")}})

        return {
            "description": property_descriptions,
            "properties": property_types,
            "required": self.required,
        }


class ToolConfig(BaseModel):
    name: str
    # type: str = "function"
    description: str
    parameters: ToolParameter

    def model_dump(self):
        parameters_dict = self.parameters.model_dump()
        parameters_descriptions_dict = parameters_dict.pop("description")
        parameters_description = ""
        for key, value in parameters_descriptions_dict.items():
            parameters_description += f"        {key}: {value}\n"

        return {
            "function_declarations": [
                {
                    "name": self.name,
                    "description": self.description + "\n\n    Args:\n" + parameters_description,
                    "parameters": {
                        "type_": "OBJECT",
                        **parameters_dict,
                    },
                }
            ]
        }


class GeminiFunctionCallingCoreModule(BaseChatModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: int = 60,
    ):
        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.max_output_tokens = max_tokens

        self.generation_config = GenerationConfig(
            stop_sequences=None,
            max_output_tokens=self.max_output_tokens,
            temperature=0.0,
            top_p=0.0,
            response_mime_type="text/plain" if not json_mode else "application/json",
        )

        self.request_options = RequestOptions(
            timeout=timeout,
        )
        self.tools = {func.__name__: func for func in tools}
        self.tool_configs = [config.model_dump() for config in tool_configs]

    def run(self, messages: list[dict[str, str]]) -> FunctionCallingResults:
        model = get_model(self.model_name, self.api_key_env_name)
        response = model.generate_content(
            contents=messages, request_options=self.request_options, tools=self.tool_configs
        )
        parts = response.candidates[0].content.parts
        results = []
        for part in parts:
            if fn := part.function_call:
                funcname = fn.name
                args = dict(fn.args)
                func_out = self.tools[funcname](**args)
                output = ToolOutput(
                    call_id=None,
                    funcname=funcname,
                    args=json.dumps(args),
                    output=func_out,
                )
                results.append(output)

        return FunctionCallingResults(
            usage=Usage(
                model_name=self.model_name,
                prompt_tokens=(model.count_tokens(messages)).total_tokens,
                completion_tokens=(model.count_tokens(parts)).total_tokens,
            ),
            results=results,
            prompt=messages,
        )

    async def arun(self, messages: list[dict[str, str]]) -> FunctionCallingResults:
        model = get_model(self.model_name, self.api_key_env_name)
        response = await model.generate_content_async(
            contents=messages, request_options=self.request_options, tools=self.tool_configs
        )
        parts = response.candidates[0].content.parts
        results = []
        for part in parts:
            if fn := part.function_call:
                funcname = fn.name
                args = dict(fn.args)
                func_out = self.tools[funcname](**args)
                output = ToolOutput(
                    call_id=None,
                    funcname=funcname,
                    args=json.dumps(args),
                    output=func_out,
                )
                results.append(output)

        return FunctionCallingResults(
            usage=Usage(
                model_name=self.model_name,
                prompt_tokens=(await model.count_tokens_async(messages)).total_tokens,
                completion_tokens=(await model.count_tokens_async(parts)).total_tokens,
            ),
            results=results,
            prompt=messages,
        )


class GeminiFunctionCallingModule(FunctionCallingWrapperModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: int = 60,
        content_filter: Optional[BaseFilter] = None,
        conversation_memory: Optional[BaseConversationMemory] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        function_calling_model = GeminiFunctionCallingCoreModule(
            api_key_env_name=api_key_env_name,
            model_name=model_name,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
            tools=tools,
            tool_configs=tool_configs,
        )

        super().__init__(
            function_calling_model=function_calling_model,
            content_filter=content_filter,
            conversation_memory=conversation_memory,
            token_counter=token_counter,
        )

    def _get_client_message_type(self) -> type[BaseMessage]:
        return GeminiMessage
