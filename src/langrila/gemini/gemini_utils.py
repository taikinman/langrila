from typing import Any, Sequence

from google.auth import credentials as auth_credentials


def get_model(
    model_name: str,
    max_output_tokens: int = 2048,
    json_mode: bool = False,
    api_key_env_name: str | None = None,
    api_type: str = "gemini",
    project_id_env_name: str | None = None,
    location_env_name: str | None = None,
    system_instruction: str | None = None,
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
    n_results: int = 1,
):
    if api_type == "genai":
        from .genai.client import get_genai_model

        return get_genai_model(
            model_name=model_name,
            api_key_env_name=api_key_env_name,
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
            response_schema=response_schema,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n_results=n_results,
        )
    elif api_type == "vertexai":
        from .vertexai.client import get_vertexai_model

        return get_vertexai_model(
            model_name=model_name,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            json_mode=json_mode,
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
            n_results=n_results,
        )
    else:
        raise ValueError(f"Unknown API type: {api_type}")


def get_message_cls(api_type: str):
    if api_type == "genai":
        from .genai.message import GeminiMessage

        return GeminiMessage
    elif api_type == "vertexai":
        from .vertexai.message import VertexAIMessage

        return VertexAIMessage
    else:
        raise ValueError(f"Unknown API type: {api_type}")


def get_tool_cls(api_type: str):
    if api_type == "genai":
        from google.ai.generativelanguage import Tool

        return Tool

    elif api_type == "vertexai":
        from vertexai.generative_models import Tool

        return Tool
    else:
        raise ValueError(f"Unknown API type: {api_type}")


def get_client_tool_type(api_type: str):
    if api_type == "genai":
        from .genai.tools import GeminiToolConfig

        return GeminiToolConfig
    elif api_type == "vertexai":
        from .vertexai.tools import VertexAIToolConfig

        return VertexAIToolConfig
    else:
        raise ValueError(f"Unknown API type: {api_type}")


def get_call_config(api_type: str, tool_choice: str | None = "auto"):
    if api_type == "genai":
        from google.ai.generativelanguage import FunctionCallingConfig, ToolConfig

        if tool_choice is None:
            return ToolConfig(
                function_calling_config=FunctionCallingConfig(mode=FunctionCallingConfig.Mode.NONE)
            )
        elif tool_choice == "auto":
            return ToolConfig(
                function_calling_config=FunctionCallingConfig(mode=FunctionCallingConfig.Mode.AUTO)
            )
        else:
            return ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfig.Mode.ANY,
                    allowed_function_names=[tool_choice],
                )
            )
    elif api_type == "vertexai":
        from vertexai.preview.generative_models import ToolConfig

        if tool_choice is None:
            return ToolConfig(
                function_calling_config=ToolConfig.FunctionCallingConfig(
                    mode=ToolConfig.FunctionCallingConfig.Mode.NONE
                )
            )
        elif tool_choice == "auto":
            return ToolConfig(
                function_calling_config=ToolConfig.FunctionCallingConfig(
                    mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
                )
            )
        else:
            return ToolConfig(
                function_calling_config=ToolConfig.FunctionCallingConfig(
                    mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    allowed_function_names=[tool_choice],
                )
            )
    else:
        raise ValueError(f"Unknown API type: {api_type}")
