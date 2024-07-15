import os
from typing import Optional

import google.generativeai as genai
from google.generativeai.types.content_types import ContentType
from google.generativeai.types.generation_types import GenerationConfig


def get_genai_model(
    model_name: str,
    api_key_env_name: str,
    system_instruction: Optional[ContentType] = None,
    max_output_tokens: int = 2048,
    json_mode: bool = False,
):
    if api_key_env_name is None:
        raise ValueError("api_key_env_name must be provided to use the Gemini API.")

    generation_config = GenerationConfig(
        stop_sequences=None,
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        top_p=0.0,
        response_mime_type="text/plain" if not json_mode else "application/json",
    )

    genai.configure(api_key=os.getenv(api_key_env_name))
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
        generation_config=generation_config,
    )
