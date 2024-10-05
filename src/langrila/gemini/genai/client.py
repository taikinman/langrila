import os
from typing import Any, Optional

import google.generativeai as genai
from google.generativeai.types import generation_types
from google.generativeai.types.content_types import ContentType
from google.generativeai.types.generation_types import GenerationConfig
from typing_extensions import override

from ...base import BaseClient
from ...utils import create_parameters


def get_genai_model(
    model_name: str,
    api_key_env_name: str,
    n_results: int = 1,
    system_instruction: Optional[ContentType] = None,
    max_output_tokens: int = 2048,
    json_mode: bool = False,
    response_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
):
    if api_key_env_name is None:
        raise ValueError("api_key_env_name must be provided to use the Gemini API.")

    generation_config = GenerationConfig(
        candidate_count=n_results,
        stop_sequences=None,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        response_mime_type="text/plain" if not json_mode else "application/json",
        response_schema=response_schema,
    )

    genai.configure(api_key=os.getenv(api_key_env_name))
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
        generation_config=generation_config,
    )


class GeminiAIStudioChat(BaseClient):
    __doc__ = genai.configure.__doc__

    def __init__(
        self,
        **kwargs,
    ):
        self.configure_params = create_parameters(genai.configure, **kwargs)

    @override
    def generate_content(self, **kwargs) -> generation_types.GenerateContentResponse:
        genai.configure(**self.configure_params)

        generation_params = create_parameters(genai.GenerativeModel.generate_content, **kwargs)

        model = genai.GenerativeModel(model_name=kwargs.get("model_name"))
        return model.generate_content(**generation_params)

    @override
    async def generate_content_async(
        self, **kwargs
    ) -> generation_types.AsyncGenerateContentResponse:
        genai.configure(**self.configure_params)

        generation_params = create_parameters(genai.GenerativeModel.generate_content, **kwargs)

        model = genai.GenerativeModel(model_name=kwargs.get("model_name"))
        return await model.generate_content_async(**generation_params)
