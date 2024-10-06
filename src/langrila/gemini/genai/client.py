import os
from typing import Any, Optional

import google.generativeai as genai
from google.generativeai.types import generation_types
from google.generativeai.types.content_types import ContentType
from google.generativeai.types.generation_types import GenerationConfig
from typing_extensions import override

from ...base import BaseClient
from ...utils import create_parameters


class GeminiAIStudioClient(BaseClient):
    def __init__(
        self,
        **kwargs,
    ):
        self.configure_params = create_parameters(genai.configure, **kwargs)

    @override
    def generate_message(self, **kwargs) -> generation_types.GenerateContentResponse:
        genai.configure(**self.configure_params)

        generation_config_params = create_parameters(GenerationConfig, **kwargs)
        generation_config = GenerationConfig(**generation_config_params)

        generation_params = create_parameters(
            genai.GenerativeModel.generate_content,
            exclude=["generation_config", "safety_config"],
            **kwargs,
        )

        model = genai.GenerativeModel(model_name=kwargs.get("model_name"))
        return model.generate_content(
            generation_config=generation_config,
            **generation_params,
        )

    @override
    async def generate_message_async(
        self, **kwargs
    ) -> generation_types.AsyncGenerateContentResponse:
        genai.configure(**self.configure_params)

        generation_config_params = create_parameters(GenerationConfig, **kwargs)
        generation_config = GenerationConfig(**generation_config_params)

        generation_params = create_parameters(
            genai.GenerativeModel.generate_content_async,
            exclude=["generation_config", "safety_config"],
            **kwargs,
        )

        model = genai.GenerativeModel(model_name=kwargs.get("model_name"))
        return await model.generate_content_async(
            generation_config=generation_config,
            **generation_params,
        )

    def count_tokens(self, **kwargs) -> int:
        genai.configure(**self.configure_params)

        model = genai.GenerativeModel(model_name=kwargs.get("model_name"))
        count_tokens_params = create_parameters(
            genai.GenerativeModel.count_tokens,
            exclude=["model_name"],
            **kwargs,
        )
        return model.count_tokens(**count_tokens_params)

    async def count_tokens_async(self, **kwargs) -> int:
        genai.configure(**self.configure_params)

        model = genai.GenerativeModel(model_name=kwargs.get("model_name"))
        count_tokens_params = create_parameters(
            genai.GenerativeModel.count_tokens_async,
            exclude=["model_name"],
            **kwargs,
        )
        return await model.count_tokens_async(**count_tokens_params)
