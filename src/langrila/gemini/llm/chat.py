import copy
import os
from typing import Any, AsyncGenerator, Generator, Optional, Sequence

from google.auth import credentials as auth_credentials

from ...base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseMessage,
)
from ...llm_wrapper import ChatWrapperModule
from ...result import CompletionResults
from ...usage import TokenCounter, Usage
from ..gemini_utils import get_client, get_message_cls


class GeminiChatCoreModule(BaseChatModule):
    def __init__(
        self,
        model_name: str,
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
        response_schema: dict[str, Any] | None = None,
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
        self.response_schema = response_schema
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

        self.system_instruction = system_instruction

        self._client = get_client(
            api_key_env_name=api_key_env_name,
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
        )

    def run(
        self, messages: list[dict[str, str]], n_results: int | None = None
    ) -> CompletionResults:
        if n_results is not None and (self.api_type == "genai" and n_results > 1):
            raise ValueError("n_results > 1 is not supported for Google AI API")

        response = self._client.generate_message(
            contents=messages,
            model_name=self.model_name,
            system_instruction=self.system_instruction,
            candidate_count=n_results,
            stop_sequences=None,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            response_mime_type="text/plain" if not self.json_mode else "application/json",
            response_schema=self.response_schema,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            **self.additional_kwargs,
        )

        usage_metadata = response.usage_metadata
        parts = []
        candidates = response.candidates
        for candidate in candidates:
            content = candidate.content
            parts.extend(content.parts)

        if self.api_type == "genai":
            from google.ai.generativelanguage import Content

        else:
            from vertexai.generative_models import Content

        content = Content(role="model", parts=parts)

        return CompletionResults(
            message=content,
            usage=Usage(
                model_name=self.model_name,
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            prompt=copy.deepcopy(messages),
        )

    async def arun(
        self, messages: list[dict[str, str]], n_results: int | None = None
    ) -> CompletionResults:
        if n_results is not None and (self.api_type == "genai" and n_results > 1):
            raise ValueError("n_results > 1 is not supported for Google AI API")

        response = await self._client.generate_message_async(
            contents=messages,
            model_name=self.model_name,
            system_instruction=self.system_instruction,
            candidate_count=n_results,
            stop_sequences=None,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            response_mime_type="text/plain" if not self.json_mode else "application/json",
            response_schema=self.response_schema,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            **self.additional_kwargs,
        )

        usage_metadata = response.usage_metadata
        parts = []
        candidates = response.candidates
        for candidate in candidates:
            content = candidate.content
            parts.extend(content.parts)

        if self.api_type == "genai":
            from google.ai.generativelanguage import Content

        else:
            from vertexai.generative_models import Content

        content = Content(role="model", parts=parts)

        return CompletionResults(
            message=content,
            usage=Usage(
                model_name=self.model_name,
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            prompt=copy.deepcopy(messages),
        )

    def stream(
        self, messages: list[dict[str, str | list[str]]]
    ) -> Generator[CompletionResults, None, None]:
        responses = self._client.generate_message(
            contents=messages,
            model_name=self.model_name,
            system_instruction=self.system_instruction,
            stop_sequences=None,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            response_mime_type="text/plain" if not self.json_mode else "application/json",
            response_schema=self.response_schema,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            stream=True,
            **self.additional_kwargs,
        )

        chunk_all = ""
        for response in responses:
            usage_metadata = response.usage_metadata
            content = response.candidates[0].content
            if content.parts[0].text:
                chunk_all += content.parts[0].text
                if hasattr(content.parts[0], "_raw_part"):
                    content.parts[0]._raw_part.text = chunk_all
                else:
                    content.parts[0].text = chunk_all

                last_content = content
                result = CompletionResults(
                    message=content,
                    usage=Usage(model_name=self.model_name),
                    prompt="",
                )

                yield result

        # at the end of the stream, return the entire response
        yield CompletionResults(
            message=last_content,
            usage=Usage(
                model_name=self.model_name,
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            prompt=copy.deepcopy(messages),
        )

    async def astream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[CompletionResults, None]:
        responses = await self._client.generate_message_async(
            contents=messages,
            model_name=self.model_name,
            system_instruction=self.system_instruction,
            stop_sequences=None,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            response_mime_type="text/plain" if not self.json_mode else "application/json",
            response_schema=self.response_schema,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            stream=True,
            **self.additional_kwargs,
        )

        chunk_all = ""
        async for _response in responses:
            usage_metadata = _response.usage_metadata
            content = _response.candidates[0].content
            if content.parts[0].text:
                chunk_all += content.parts[0].text
                if hasattr(content.parts[0], "_raw_part"):
                    content.parts[0]._raw_part.text = chunk_all
                else:
                    content.parts[0].text = chunk_all

                last_content = content

                result = CompletionResults(
                    message=content,
                    usage=Usage(model_name=self.model_name),
                    prompt="",
                )

                yield result

        # at the end of the stream, return the entire response
        yield CompletionResults(
            message=last_content,
            usage=Usage(
                model_name=self.model_name,
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            prompt=copy.deepcopy(messages),
        )


class GeminiChatModule(ChatWrapperModule):
    def __init__(
        self,
        model_name: str,
        api_key_env_name: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: int = 60,
        content_filter: Optional[BaseFilter] = None,
        conversation_memory: Optional[BaseConversationMemory] = None,
        system_instruction: str | None = None,
        token_counter: TokenCounter | None = None,
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
        response_schema: dict[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        # The module to call client API
        chat_model = GeminiChatCoreModule(
            api_key_env_name=api_key_env_name,
            model_name=model_name,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
            system_instruction=system_instruction,
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
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        super().__init__(
            chat_model=chat_model,
            content_filter=content_filter,
            conversation_memory=conversation_memory,
            token_counter=token_counter,
        )

    def _get_client_message_type(self) -> type[BaseMessage]:
        return get_message_cls(self.chat_model.api_type)
