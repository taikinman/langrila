import os
from typing import Sequence

from google.auth import credentials as auth_credentials


def get_client(
    api_key_env_name: str | None = None,
    api_type: str = "gemini",
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
):
    if api_type == "genai":
        from .genai.client import GeminiAIStudioClient

        return GeminiAIStudioClient(
            api_key=os.getenv(api_key_env_name),
        )

    else:
        from .vertexai.client import GeminiVertexAIClient

        return GeminiVertexAIClient(
            project=os.getenv(project_id_env_name),
            location=os.getenv(location_env_name),
            experiment=experiment,
            experiment_description=experiment_description,
            experiment_tensorboard=experiment_tensorboard,
            staging_bucket=staging_bucket,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            network=network,
            service_account=service_account,
            api_endpoint=os.getenv(endpoint_env_name) if endpoint_env_name else None,
            request_metadata=request_metadata,
            api_key=os.getenv(api_key_env_name) if api_key_env_name else None,
        )


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


def merge_responses(response, api_type: str):
    parts = []
    candidates = response.candidates
    for candidate in candidates:
        content = candidate.content
        parts.extend(content.parts)

    if api_type == "genai":
        from google.ai.generativelanguage import Content

    else:
        from vertexai.generative_models import Content

    return Content(role="model", parts=parts)
