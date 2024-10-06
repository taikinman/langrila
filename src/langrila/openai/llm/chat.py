from copy import deepcopy
from typing import Any, AsyncGenerator, Generator, Literal, Mapping

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.lib.azure import AzureADTokenProvider
from openai.types.chat import ChatCompletionAssistantMessageParam
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass

from ...base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
)
from ...llm_wrapper import ChatWrapperModule
from ...result import CompletionResults
from ...usage import TokenCounter, Usage
from ...warnings import change_function
from ..conversation_adjuster.truncate import OldConversationTruncationModule
from ..message import OpenAIMessage
from ..openai_utils import get_client, get_n_tokens, get_token_limit


class OpenAIChatCoreModule(BaseChatModule):
    @change_function("temperature", "run()/arun()/stream()/astream()")
    @change_function("max_tokens", "run()/arun()/stream()/astream()")
    @change_function("max_completion_tokens", "run()/arun()/stream()/astream()")
    @change_function("top_p", "run()/arun()/stream()/astream()")
    @change_function("frequency_penalty", "run()/arun()/stream()/astream()")
    @change_function("presence_penalty", "run()/arun()/stream()/astream()")
    @change_function("user", "run()/arun()/stream()/astream()")
    @change_function("seed", "run()/arun()/stream()/astream()")
    @change_function("response_format", "run()/arun()/stream()/astream()")
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        organization_id_env_name: str | None = None,
        api_type: Literal["openai", "azure"] = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        timeout: int = 60,
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

        self.additional_inputs = {}
        self.seed = seed
        self.additional_inputs["seed"] = seed
        if json_mode:
            if response_schema:
                self.response_format = response_schema
                self.additional_inputs["response_format"] = self.response_format
            else:
                self.response_format = {"type": "json_object"} if json_mode else NOT_GIVEN
                self.additional_inputs["response_format"] = self.response_format
        else:
            self.response_format = NOT_GIVEN
            self.additional_inputs["response_format"] = self.response_format

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
        self, messages: list[dict[str, str]], n_results: int | NotGiven = NOT_GIVEN
    ) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        token_arg = {}
        if self.max_tokens is not NOT_GIVEN:
            token_arg["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not NOT_GIVEN:
            token_arg["max_completion_tokens"] = self.max_completion_tokens

        response = self._client.generate_message(
            model=self.model_name,
            messages=_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=NOT_GIVEN,
            n=n_results,
            user=self.user,
            **self.additional_inputs,
            **token_arg,
        )

        usage = Usage(model_name=self.model_name)
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
        self, messages: list[dict[str, str]], n_results: int | NotGiven = NOT_GIVEN
    ) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        token_arg = {}
        if self.max_tokens is not NOT_GIVEN:
            token_arg["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not NOT_GIVEN:
            token_arg["max_completion_tokens"] = self.max_completion_tokens

        response = await self._client.generate_message_async(
            model=self.model_name,
            messages=_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=NOT_GIVEN,
            n=n_results,
            user=self.user,
            **self.additional_inputs,
            **token_arg,
        )

        usage = Usage(model_name=self.model_name)
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
        self, messages: list[dict[str, str]], stream_options: dict[str, Any] | None = None
    ) -> Generator[CompletionResults, None, None]:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        if stream_options is None:
            stream_options = {"stream_options": {"include_usage": True}}
        else:
            stream_options = {"stream_options": stream_options | {"include_usage": True}}

        additional_inputs = self.additional_inputs | stream_options

        token_arg = {}
        if self.max_tokens is not NOT_GIVEN:
            token_arg["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not NOT_GIVEN:
            token_arg["max_completion_tokens"] = self.max_completion_tokens

        response = self._client.generate_message(
            model=self.model_name,
            messages=_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=NOT_GIVEN,
            user=self.user,
            stream=True,
            **additional_inputs,
            **token_arg,
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
                            usage=Usage(model_name=self.model_name),
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
            model_name=self.model_name,
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
        self, messages: list[dict[str, str]], stream_options: dict[str, Any] | None = None
    ) -> AsyncGenerator[CompletionResults, None]:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        if stream_options is None:
            stream_options = {"stream_options": {"include_usage": True}}
        else:
            stream_options = {"stream_options": stream_options | {"include_usage": True}}

        additional_inputs = self.additional_inputs | stream_options

        token_arg = {}
        if self.max_tokens is not NOT_GIVEN:
            token_arg["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not NOT_GIVEN:
            token_arg["max_completion_tokens"] = self.max_completion_tokens

        response = await self._client.generate_message_async(
            model=self.model_name,
            messages=_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=NOT_GIVEN,
            user=self.user,
            stream=True,
            **additional_inputs,
            **token_arg,
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
                            usage=Usage(model_name=self.model_name),
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
            model_name=self.model_name,
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
        model_name: str,
        api_type: str = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        organization_id_env_name: str | None = None,
        max_tokens: int | None = 2048,
        timeout: int = 60,
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
    ):
        # The module to call client API
        chat_model = OpenAIChatCoreModule(
            api_key_env_name=api_key_env_name,
            organization_id_env_name=organization_id_env_name,
            model_name=model_name,
            max_tokens=max_tokens,
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
        )

        super().__init__(
            chat_model=chat_model,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
        )

    def _get_client_message_type(self) -> OpenAIMessage:
        return OpenAIMessage
