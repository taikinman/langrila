import os
from typing import Any, AsyncIterable, Iterable, Sequence

import vertexai
from google.auth import credentials as auth_credentials
from google.cloud.aiplatform.tensorboard import tensorboard_resource
from typing_extensions import override
from vertexai.generative_models import GenerationConfig, GenerationResponse, GenerativeModel

from ...base import BaseClient
from ...utils import create_parameters


class GeminiVertexAIClient(BaseClient):
    def __init__(self, **kwargs):
        self.configure_params = create_parameters(vertexai.init, **kwargs)

    @override
    def generate_message(self, **kwargs) -> GenerationResponse | Iterable[GenerationResponse]:
        vertexai.init(**self.configure_params)

        generation_config_params = create_parameters(GenerationConfig, **kwargs)
        generation_config = GenerationConfig(**generation_config_params)

        generation_params = create_parameters(GenerativeModel.generate_content, **kwargs)

        model = GenerativeModel(
            model_name=kwargs.get("model_name"), system_instruction=kwargs.get("system_instruction")
        )

        self._warn_ignore_params(
            kwargs,
            {
                **generation_config_params,
                **generation_params,
                "model_name": kwargs.get("model_name"),
                "system_instruction": kwargs.get("system_instruction"),
            },
        )

        return model.generate_content(generation_config=generation_config, **generation_params)

    @override
    async def generate_message_async(
        self, **kwargs
    ) -> GenerationResponse | AsyncIterable[GenerationResponse]:
        vertexai.init(**self.configure_params)

        generation_config_params = create_parameters(GenerationConfig, **kwargs)
        generation_config = GenerationConfig(**generation_config_params)

        generation_params = create_parameters(GenerativeModel.generate_content_async, **kwargs)

        model = GenerativeModel(
            model_name=kwargs.get("model_name"), system_instruction=kwargs.get("system_instruction")
        )

        self._warn_ignore_params(
            kwargs,
            {
                **generation_config_params,
                **generation_params,
                "model_name": kwargs.get("model_name"),
                "system_instruction": kwargs.get("system_instruction"),
            },
        )

        return await model.generate_content_async(
            generation_config=generation_config,
            **generation_params,
        )

    def count_tokens(self, **kwargs) -> int:
        vertexai.init(**self.configure_params)

        model = GenerativeModel(model_name=kwargs.get("model_name"))
        count_tokens_params = create_parameters(
            GenerativeModel.count_tokens,
            exclude=["model_name"],
            **kwargs,
        )

        return model.count_tokens(**count_tokens_params)

    async def count_tokens_async(self, **kwargs) -> int:
        vertexai.init(**self.configure_params)

        model = GenerativeModel(model_name=kwargs.get("model_name"))
        count_tokens_params = create_parameters(
            GenerativeModel.count_tokens_async,
            exclude=["model_name"],
            **kwargs,
        )
        return await model.count_tokens_async(**count_tokens_params)
