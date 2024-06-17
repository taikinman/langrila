import os
from typing import Optional

import google.generativeai as genai
from google.generativeai.types.content_types import ContentType, FunctionLibraryType, ToolConfigType


def get_model(
    model_name: str,
    api_key_env_name: str,
    system_instruction: Optional[ContentType] = None,
    tools: Optional[FunctionLibraryType] = None,
    tool_config: Optional[ToolConfigType] = None,
):
    genai.configure(api_key=os.getenv(api_key_env_name))
    return genai.GenerativeModel(
        model_name=model_name,
        tools=tools,
        tool_config=tool_config,
        system_instruction=system_instruction,
    )
