import copy
import json
from typing import Any, Callable, Optional, Sequence

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
from ...result import FunctionCallingResults, ToolCallResponse, ToolOutput
from ...tools import ToolConfig
from ...usage import TokenCounter, Usage
from ...utils import generate_dummy_call_id
from ..gemini_utils import (
    get_call_config,
    get_client_tool_type,
    get_message_cls,
    get_model,
    get_tool_cls,
)


class GeminiFunctionCallingCoreModule(BaseFunctionCallingModule):
    def __init__(
        self,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
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
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: int = 60,
        system_instruction: str | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.max_output_tokens = max_tokens
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
        self.json_mode = json_mode
        self.system_instruction = system_instruction
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.additional_kwargs = {}
        if api_type == "genai":
            from google.generativeai.types.helper_types import RequestOptions

            request_options = RequestOptions(
                timeout=timeout,
            )
            self.additional_kwargs["request_options"] = request_options

        ClientToolConfig = self._get_client_tool_config_type(api_type)
        client_tool_configs = ClientToolConfig.from_universal_configs(tool_configs)
        self.tools = self._set_runnable_tools_dict(tools)

        tool_cls = get_tool_cls(api_type=api_type)
        function_declarations = [config.format() for config in client_tool_configs]
        self.tool_configs = [tool_cls(function_declarations=function_declarations)]

    def _get_call_config(self, tool_choice: str | None = "auto"):
        return get_call_config(api_type=self.api_type, tool_choice=tool_choice)

    def _get_client_tool_config_type(self, api_type: str):
        return get_client_tool_type(api_type=api_type)

    def run(
        self, messages: list[dict[str, str]], tool_choice: list[str] | str | None = "auto"
    ) -> FunctionCallingResults:
        model = get_model(
            model_name=self.model_name,
            api_key_env_name=self.api_key_env_name,
            max_output_tokens=self.max_output_tokens,
            json_mode=self.json_mode,
            system_instruction=self.system_instruction,
            api_type=self.api_type,
            project_id_env_name=self.project_id_env_name,
            location_env_name=self.location_env_name,
            experiment=self.experiment,
            experiment_description=self.experiment_description,
            experiment_tensorboard=self.experiment_tensorboard,
            staging_bucket=self.staging_bucket,
            credentials=self.credentials,
            encryption_spec_key_name=self.encryption_spec_key_name,
            network=self.network,
            service_account=self.service_account,
            endpoint_env_name=self.endpoint_env_name,
            request_metadata=self.request_metadata,
            n_results=1,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        call_config = self._get_call_config(tool_choice=tool_choice)

        response = model.generate_content(
            contents=messages,
            tools=self.tool_configs,
            tool_config=call_config,
            **self.additional_kwargs,
        )
        parts = response.candidates[0].content.parts

        results = []
        calls = []
        for part in parts:
            if fn := part.function_call:
                funcname = fn.name
                args = dict(fn.args)
                func_out = self.tools[funcname](**args)
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
                model_name=self.model_name,
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            results=results,
            calls=calls,
            prompt=copy.deepcopy(messages),
        )

    async def arun(
        self, messages: list[dict[str, str]], tool_choice: list[str] | str | None = "auto"
    ) -> FunctionCallingResults:
        model = get_model(
            model_name=self.model_name,
            api_key_env_name=self.api_key_env_name,
            max_output_tokens=self.max_output_tokens,
            json_mode=self.json_mode,
            system_instruction=self.system_instruction,
            api_type=self.api_type,
            project_id_env_name=self.project_id_env_name,
            location_env_name=self.location_env_name,
            experiment=self.experiment,
            experiment_description=self.experiment_description,
            experiment_tensorboard=self.experiment_tensorboard,
            staging_bucket=self.staging_bucket,
            credentials=self.credentials,
            encryption_spec_key_name=self.encryption_spec_key_name,
            network=self.network,
            service_account=self.service_account,
            endpoint_env_name=self.endpoint_env_name,
            request_metadata=self.request_metadata,
            n_results=1,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        call_config = self._get_call_config(tool_choice=tool_choice)

        response = await model.generate_content_async(
            contents=messages,
            tools=self.tool_configs,
            tool_config=call_config,
            **self.additional_kwargs,
        )
        parts = response.candidates[0].content.parts

        results = []
        calls = []
        for part in parts:
            if fn := part.function_call:
                funcname = fn.name
                args = dict(fn.args)
                func_out = self.tools[funcname](**args)
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
                model_name=self.model_name,
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            results=results,
            calls=calls,
            prompt=copy.deepcopy(messages),
        )


class GeminiFunctionCallingModule(FunctionCallingWrapperModule):
    def __init__(
        self,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        api_key_env_name: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: int = 60,
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
    ):
        # The module to call client API
        function_calling_model = GeminiFunctionCallingCoreModule(
            api_key_env_name=api_key_env_name,
            model_name=model_name,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
            tools=tools,
            tool_configs=tool_configs,
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
            system_instruction=system_instruction,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        super().__init__(
            function_calling_model=function_calling_model,
            content_filter=content_filter,
            conversation_memory=conversation_memory,
            token_counter=token_counter,
        )

    def _get_client_message_type(self) -> type[BaseMessage]:
        return get_message_cls(self.function_calling_model.api_type)
