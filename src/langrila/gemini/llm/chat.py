import copy
import os
from typing import Any, AsyncGenerator, Generator, Iterable, Mapping, Optional, Sequence

from google.auth import credentials as auth_credentials

from ...base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseMessage,
)
from ...llm_wrapper import ChatWrapperModule
from ...message_content import ConversationType, InputType
from ...result import CompletionResults
from ...usage import TokenCounter, Usage
from ..gemini_utils import get_client, get_message_cls, merge_responses


class GeminiChatCoreModule(BaseChatModule):
    def __init__(
        self,
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
        **kwargs: Any,
    ):
        self.api_type = api_type
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
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> CompletionResults:
        response = self._client.generate_message(
            contents=messages,
            **kwargs,
        )

        usage_metadata = response.usage_metadata
        content = merge_responses(response, api_type=self.api_type)

        return CompletionResults(
            message=content,
            usage=Usage(
                model_name=kwargs.get("model_name"),
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            prompt=copy.deepcopy(messages),
        )

    async def arun(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> CompletionResults:
        response = await self._client.generate_message_async(
            contents=messages,
            **kwargs,
        )

        usage_metadata = response.usage_metadata
        content = merge_responses(response, api_type=self.api_type)

        return CompletionResults(
            message=content,
            usage=Usage(
                model_name=kwargs.get("model_name"),
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            prompt=copy.deepcopy(messages),
        )

    def stream(
        self,
        messages: list[dict[str, str | list[str]]],
        **kwargs: Any,
    ) -> Generator[CompletionResults, None, None]:
        responses = self._client.generate_message(
            contents=messages,
            stream=True,
            **kwargs,
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
                    usage=Usage(model_name=kwargs.get("model_name")),
                    prompt="",
                )

                yield result

        # at the end of the stream, return the entire response
        yield CompletionResults(
            message=last_content,
            usage=Usage(
                model_name=kwargs.get("model_name"),
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            prompt=copy.deepcopy(messages),
        )

    async def astream(
        self,
        messages: list[dict[str, str | list[str]]],
        **kwargs: Any,
    ) -> AsyncGenerator[CompletionResults, None]:
        responses = await self._client.generate_message_async(
            contents=messages,
            stream=True,
            **kwargs,
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
                    usage=Usage(model_name=kwargs.get("model_name")),
                    prompt="",
                )

                yield result

        # at the end of the stream, return the entire response
        yield CompletionResults(
            message=last_content,
            usage=Usage(
                model_name=kwargs.get("model_name"),
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
        max_output_tokens: int | None = None,
        json_mode: bool = False,
        timeout: int | None = None,
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
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        response_mime_type: str | None = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_key_env_name = api_key_env_name
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        self.timeout = timeout
        self.system_instruction = system_instruction
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
        self.response_schema = response_schema
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.routing_config = routing_config
        self.logprobs = logprobs
        self.response_logprobs = response_logprobs
        self.response_mime_type = response_mime_type

        # The module to call client API
        chat_model = GeminiChatCoreModule(
            api_key_env_name=api_key_env_name,
            model_name=model_name,
            max_output_tokens=max_output_tokens,
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
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            response_mime_type=response_mime_type,
            **kwargs,
        )

        super().__init__(
            chat_model=chat_model,
            content_filter=content_filter,
            conversation_memory=conversation_memory,
            token_counter=token_counter,
        )

    def _get_generation_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        _kwargs = {}
        _kwargs["system_instruction"] = kwargs.get("system_instruction") or self.system_instruction
        _kwargs["model_name"] = kwargs.get("model_name") or self.model_name
        _kwargs["temperature"] = kwargs.get("temperature") or self.temperature
        _kwargs["top_p"] = kwargs.get("top_p") or self.top_p
        _kwargs["top_k"] = kwargs.get("top_k") or self.top_k
        _kwargs["stop_sequences"] = kwargs.get("stop_sequences")
        _kwargs["frequency_penalty"] = kwargs.get("frequency_penalty ") or self.frequency_penalty
        _kwargs["presence_penalty"] = kwargs.get("presence_penalty") or self.presence_penalty
        _kwargs["seed"] = kwargs.get("seed") or self.seed
        _kwargs["response_mime_type"] = (
            "application/json"
            if kwargs.get("json_mode")
            else kwargs.get("response_mime_type") or self.response_mime_type
        )
        _kwargs["response_schema"] = kwargs.get("response_schema") or self.response_schema
        _kwargs["max_output_tokens"] = kwargs.get("max_output_tokens") or self.max_output_tokens
        _kwargs["routing_config"] = kwargs.get("routing_config") or self.routing_config
        _kwargs["logprobs"] = kwargs.get("logprobs") or self.logprobs
        _kwargs["response_logprobs"] = kwargs.get("response_logprobs") or self.response_logprobs
        _kwargs["candidate_count"] = kwargs.get("n_results") or 1

        if self.api_type == "genai":
            from google.generativeai.types.helper_types import RequestOptions

            _kwargs["request_options"] = RequestOptions(
                timeout=kwargs.get("timeout") or 60,
            )

        return _kwargs

    def _get_client_message_type(self) -> type[BaseMessage]:
        return get_message_cls(self.api_type)

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        system_instruction: str | None = None,
        model_name: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_mime_type: str | None = None,
        json_mode: bool = False,
        response_schema: Mapping[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        n_results: int | None = None,
        **kwargs: Any,
    ) -> CompletionResults:
        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_mime_type=response_mime_type,
            json_mode=json_mode,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            n_results=n_results,
            **kwargs,
        )
        return super().run(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        system_instruction: str | None = None,
        model_name: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_mime_type: str | None = None,
        json_mode: bool = False,
        response_schema: Mapping[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        n_results: int | None = None,
        **kwargs: Any,
    ) -> CompletionResults:
        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_mime_type=response_mime_type,
            json_mode=json_mode,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            n_results=n_results,
            **kwargs,
        )

        return await super().arun(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        system_instruction: str | None = None,
        model_name: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_mime_type: str | None = None,
        json_mode: bool = False,
        response_schema: Mapping[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        **kwargs: Any,
    ) -> Generator[CompletionResults, None, None]:
        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_mime_type=response_mime_type,
            json_mode=json_mode,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            **kwargs,
        )

        return super().stream(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        system_instruction: str | None = None,
        model_name: str | None = None,
        stop_sequences: Iterable[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_mime_type: str | None = None,
        json_mode: bool = False,
        response_schema: Mapping[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        routing_config: Any | None = None,
        logprobs: int | None = None,
        response_logprobs: bool | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[CompletionResults, None]:
        generation_kwargs = self._get_generation_kwargs(
            system_instruction=system_instruction,
            model_name=model_name,
            stop_sequences=stop_sequences,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_mime_type=response_mime_type,
            json_mode=json_mode,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            routing_config=routing_config,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            **kwargs,
        )

        return super().astream(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )
