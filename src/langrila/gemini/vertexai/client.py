import os
from typing import Any, Callable, Optional, Sequence

import vertexai
from google.auth import credentials as auth_credentials
from google.cloud.aiplatform.tensorboard import tensorboard_resource
from vertexai.generative_models import GenerationConfig, GenerativeModel


def get_vertexai_model(
    model_name: str,
    project_id_env_name: str,
    location_env_name: str,
    n_results: int = 1,
    system_instruction: str | None = None,
    max_output_tokens: int = 2048,
    json_mode: bool = False,
    experiment: str | None = None,
    experiment_description: str | None = None,
    experiment_tensorboard: str | tensorboard_resource.Tensorboard | bool | None = None,
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
):
    vertexai.init(
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
        api_endpoint=os.getenv(endpoint_env_name) if endpoint_env_name else endpoint_env_name,
        request_metadata=request_metadata,
    )

    generation_config = GenerationConfig(
        candidate_count=n_results,
        stop_sequences=None,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        response_mime_type="text/plain" if not json_mode else "application/json",
        response_schema=response_schema,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )

    return GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
        generation_config=generation_config,
    )
