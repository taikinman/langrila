from copy import deepcopy
from typing import Any, AsyncGenerator, Generator, Literal, Mapping

import httpx
from openai._types import NOT_GIVEN, NotGiven
from openai.lib.azure import AzureADTokenProvider
from openai.types.chat import ChatCompletionAssistantMessageParam
from pydantic import BaseModel

from ...base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
)
from ...llm_wrapper import ChatWrapperModule
from ...message_content import ConversationType, InputType
from ...result import CompletionResults
from ...usage import TokenCounter, Usage
from ..message import OpenAIMessage
from ..openai_utils import get_client


class OpenAIChatCoreModule(BaseChatModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str | None = None,
        organization_id_env_name: str | None = None,
        api_type: Literal["openai", "azure"] = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        timeout: int | NotGiven = NOT_GIVEN,
        max_retries: int = 2,
        seed: int | NotGiven = NOT_GIVEN,
        json_mode: bool = False,
        system_instruction: str | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        top_p: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        response_schema: BaseModel | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AzureADTokenProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
        **kwargs: Any,
    ) -> None:
        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.organization_id_env_name = organization_id_env_name

        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_type = api_type
        self.api_version = api_version
        self.endpoint_env_name = endpoint_env_name
        self.deployment_id_env_name = deployment_id_env_name
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.user = user
        self.top_p = top_p

        self.seed = seed
        self.json_mode = json_mode
        self.response_schema = response_schema

        if system_instruction:
            system_instruction = OpenAIMessage.to_universal_message(
                role="system", message=system_instruction
            )
            self.system_instruction = OpenAIMessage.to_client_message(system_instruction)
        else:
            self.system_instruction = None

        self.conversation_length_adjuster = conversation_length_adjuster

        self._client = get_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
            project=project,
            base_url=base_url,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )

    def run(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        _messages = (
            [kwargs.get("system_instruction")] + messages
            if kwargs.get("system_instruction")
            else messages
        )
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        response = self._client.generate_message(
            messages=_messages,
            **kwargs,
        )

        usage = Usage(model_name=kwargs.get("model") or self.model_name)
        usage += response.usage
        choices = response.choices
        contents = []
        for choice in choices:
            response_message = choice.message.content.strip("\n")
            contents.append({"type": "text", "text": response_message})

        return CompletionResults(
            usage=usage,
            message=ChatCompletionAssistantMessageParam(role="assistant", content=contents),
            prompt=deepcopy(_messages),
        )

    async def arun(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        _messages = (
            [kwargs.get("system_instruction")] + messages
            if kwargs.get("system_instruction")
            else messages
        )

        response = await self._client.generate_message_async(
            messages=_messages,
            **kwargs,
        )

        usage = Usage(model_name=kwargs.get("model") or self.model_name)
        usage += response.usage
        choices = response.choices
        contents = []
        for choice in choices:
            response_message = choice.message.content.strip("\n")
            contents.append({"type": "text", "text": response_message})

        return CompletionResults(
            usage=usage,
            message=ChatCompletionAssistantMessageParam(role="assistant", content=contents),
            prompt=deepcopy(_messages),
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Generator[CompletionResults, None, None]:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        _messages = (
            [kwargs.get("system_instruction")] + messages
            if kwargs.get("system_instruction")
            else messages
        )

        response = self._client.generate_message(
            messages=_messages,
            stream=True,
            **kwargs,
        )

        all_chunk = ""
        prompt_tokens = 0
        completion_tokens = 0
        for r in response:
            if len(r.choices) > 0:
                delta = r.choices[0].delta
                if delta is not None:
                    chunk = delta.content
                    if chunk is not None:
                        all_chunk += chunk
                        yield CompletionResults(
                            usage=Usage(model_name=kwargs.get("model") or self.model_name),
                            message=ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=[{"type": "text", "text": all_chunk}],
                            ),
                            prompt=[{}],
                        )

            else:
                if r.usage:
                    prompt_tokens = r.usage.prompt_tokens
                    completion_tokens = r.usage.completion_tokens

        # at the end of stream, return the whole message and usage
        usage = Usage(
            model_name=kwargs.get("model") or self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        yield CompletionResults(
            usage=usage,
            message=ChatCompletionAssistantMessageParam(
                role="assistant",
                content=[{"type": "text", "text": all_chunk}],
            ),
            prompt=deepcopy(_messages),
        )

    async def astream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[CompletionResults, None]:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        _messages = (
            [kwargs.get("system_instruction")] + messages
            if kwargs.get("system_instruction")
            else messages
        )

        if kwargs.get("stream_options") is None:
            stream_options = {"include_usage": True}
            kwargs["stream_options"] = stream_options
        else:
            kwargs["stream_options"] = {"include_usage": True, **kwargs.get("stream_options", {})}

        response = await self._client.generate_message_async(
            messages=_messages,
            stream=True,
            **kwargs,
        )

        all_chunk = ""
        prompt_tokens = 0
        completion_tokens = 0
        async for r in response:
            if len(r.choices) > 0:
                delta = r.choices[0].delta
                if delta is not None:
                    chunk = delta.content
                    if chunk is not None:
                        all_chunk += chunk
                        yield CompletionResults(
                            usage=Usage(model_name=kwargs.get("model") or self.model_name),
                            message=ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=[{"type": "text", "text": all_chunk}],
                            ),
                            prompt=[{}],
                        )

            else:
                if r.usage:
                    prompt_tokens = r.usage.prompt_tokens
                    completion_tokens = r.usage.completion_tokens

        # at the end of stream, return the whole message and usage
        usage = Usage(
            model_name=kwargs.get("model") or self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        yield CompletionResults(
            usage=usage,
            message=ChatCompletionAssistantMessageParam(
                role="assistant",
                content=[{"type": "text", "text": all_chunk}],
            ),
            prompt=deepcopy(_messages),
        )


class OpenAIChatModule(ChatWrapperModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str | None = None,
        api_type: str = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        organization_id_env_name: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        timeout: int | NotGiven = NOT_GIVEN,
        max_retries: int = 2,
        seed: int | NotGiven = NOT_GIVEN,
        json_mode: bool = False,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        system_instruction: str | None = None,
        token_counter: TokenCounter | None = None,
        top_p: float | None | NotGiven = NOT_GIVEN,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        response_schema: BaseModel | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AzureADTokenProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
        **kwargs: Any,
    ):
        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.api_type = api_type
        self.api_version = api_version
        self.endpoint_env_name = endpoint_env_name
        self.deployment_id_env_name = deployment_id_env_name
        self.organization_id_env_name = organization_id_env_name
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.seed = seed
        self.json_mode = json_mode
        self.system_instruction = system_instruction
        self.conversation_length_adjuster = conversation_length_adjuster
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.user = user
        self.response_schema = response_schema
        self.project = project
        self.base_url = base_url
        self.azure_ad_token = azure_ad_token
        self.azure_ad_token_provider = azure_ad_token_provider
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self._strict_response_validation = _strict_response_validation

        # The module to call client API
        chat_model = OpenAIChatCoreModule(
            api_key_env_name=api_key_env_name,
            organization_id_env_name=organization_id_env_name,
            model_name=model_name,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            api_type=api_type,
            api_version=api_version,
            endpoint_env_name=endpoint_env_name,
            deployment_id_env_name=deployment_id_env_name,
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
            json_mode=json_mode,
            system_instruction=system_instruction,
            conversation_length_adjuster=conversation_length_adjuster,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            response_schema=response_schema,
            project=project,
            base_url=base_url,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
            **kwargs,
        )

        super().__init__(
            chat_model=chat_model,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
        )

    def _system_instruction_to_message(self, system_instruction: str | None) -> OpenAIMessage:
        if system_instruction:
            _system_instruction = OpenAIMessage.to_universal_message(
                role="system", message=system_instruction
            )
            return OpenAIMessage.to_client_message(_system_instruction)
        else:
            return None

    def _get_client_message_type(self) -> OpenAIMessage:
        return OpenAIMessage

    def _get_response_format(
        self, json_mode: bool, response_schema: BaseModel | None
    ) -> dict[str, Any] | NotGiven:
        if json_mode:
            if response_schema:
                return {"response_format": response_schema}
            else:
                response_format = {"type": "json_object"} if json_mode else NOT_GIVEN
                return {"response_format": response_format}
        else:
            return NOT_GIVEN

    def _get_generation_kwargs(self, **kwargs: Any) -> None:
        _kwargs = {}
        _kwargs["system_instruction"] = self._system_instruction_to_message(
            kwargs.get("system_instruction") or self.system_instruction
        )
        _kwargs["model"] = kwargs.get("model_name") or self.model_name
        _kwargs["temperature"] = kwargs.get("temperature") or self.temperature
        _kwargs["top_p"] = kwargs.get("top_p") or self.top_p
        _kwargs["stop"] = kwargs.get("stop")
        _kwargs["frequency_penalty"] = kwargs.get("frequency_penalty ") or self.frequency_penalty
        _kwargs["presence_penalty"] = kwargs.get("presence_penalty") or self.presence_penalty
        _kwargs["user"] = kwargs.get("user") or self.user
        _kwargs["seed"] = kwargs.get("seed") or self.seed
        _kwargs["n"] = kwargs.get("n_results")
        _kwargs["response_format"] = self._get_response_format(
            json_mode=kwargs.get("json_mode") or self.json_mode,
            response_schema=kwargs.get("response_schema") or self.response_schema,
        )

        _kwargs["max_tokens"] = kwargs.get("max_tokens") or self.max_tokens
        _kwargs["max_completion_tokens"] = (
            kwargs.get("max_completion_tokens") or self.max_completion_tokens
        )

        return _kwargs

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        stop: str | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        system_instruction: str | None = None,
        json_mode: bool = False,
        response_schema: BaseModel | None = None,
        n_results: int | None = None,
        seed: int | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> CompletionResults:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            system_instruction=system_instruction,
            json_mode=json_mode,
            response_schema=response_schema,
            n_results=n_results,
            seed=seed,
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
        model_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        stop: str | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        system_instruction: str | None = None,
        json_mode: bool = False,
        response_schema: BaseModel | None = None,
        n_results: int | None = None,
        seed: int | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> CompletionResults:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            system_instruction=system_instruction,
            json_mode=json_mode,
            response_schema=response_schema,
            n_results=n_results,
            seed=seed,
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
        model_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        stop: str | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        stream_options: dict[str, Any] | None = None,
        system_instruction: str | None = None,
        seed: int | NotGiven = NOT_GIVEN,
        json_mode: bool = False,
        response_schema: BaseModel | None = None,
        **kwargs: Any,
    ) -> Generator[CompletionResults, None, None]:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            system_instruction=system_instruction,
            json_mode=json_mode,
            response_schema=response_schema,
            seed=seed,
            **kwargs,
        )

        if kwargs.get("stream_options") is None:
            stream_options = {"include_usage": True}
            generation_kwargs["stream_options"] = stream_options
        else:
            generation_kwargs["stream_options"] = {
                "include_usage": True,
                **kwargs.get("stream_options", {}),
            }

        return super().stream(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        stop: str | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        stream_options: dict[str, Any] | None = None,
        system_instruction: str | None = None,
        seed: int | NotGiven = NOT_GIVEN,
        json_mode: bool = False,
        response_schema: BaseModel | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[CompletionResults, None]:
        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            system_instruction=system_instruction,
            json_mode=json_mode,
            response_schema=response_schema,
            seed=seed,
            **kwargs,
        )

        if kwargs.get("stream_options") is None:
            stream_options = {"include_usage": True}
            generation_kwargs["stream_options"] = stream_options
        else:
            generation_kwargs["stream_options"] = {
                "include_usage": True,
                **kwargs.get("stream_options", {}),
            }

        return super().astream(
            prompt=prompt,
            init_conversation=init_conversation,
            **generation_kwargs,
        )
