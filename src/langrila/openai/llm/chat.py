from copy import deepcopy
from typing import Any, AsyncGenerator, Generator

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai._types import NOT_GIVEN, NotGiven
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
from ..conversation_adjuster.truncate import OldConversationTruncationModule
from ..message import OpenAIMessage
from ..model_config import _OLDER_MODEL_CONFIG, MODEL_CONFIG, MODEL_POINT
from ..openai_utils import get_async_client, get_client, get_n_tokens, get_token_limit


def completion(
    client: OpenAI | AzureOpenAI,
    model_name: str,
    messages: Any,
    max_tokens: int,
    stream: bool,
    top_p: float | NotGiven = NOT_GIVEN,
    stop: str | NotGiven = NOT_GIVEN,
    frequency_penalty: float | NotGiven = NOT_GIVEN,
    n_results: int | NotGiven = NOT_GIVEN,
    presence_penalty: float | NotGiven = NOT_GIVEN,
    temperature: float | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    **kwargs,
):
    params = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        stream=stream,
        n=n_results,
        user=user,
        **kwargs,
    )

    return client.chat.completions.create(**params)


async def acompletion(
    client: AsyncOpenAI | AsyncAzureOpenAI,
    model_name: str,
    messages: Any,
    max_tokens: int,
    stream: bool,
    top_p: float | NotGiven = NOT_GIVEN,
    stop: str | NotGiven = NOT_GIVEN,
    frequency_penalty: float | NotGiven = NOT_GIVEN,
    n_results: int | NotGiven = NOT_GIVEN,
    presence_penalty: float | NotGiven = NOT_GIVEN,
    temperature: float | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    **kwargs,
):
    params = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        stream=stream,
        n=n_results,
        user=user,
        **kwargs,
    )

    return await client.chat.completions.create(**params)


def parse(
    client: OpenAI | AzureOpenAI,
    model_name: str,
    messages: Any,
    max_tokens: int,
    top_p: float | NotGiven = NOT_GIVEN,
    stop: str | NotGiven = NOT_GIVEN,
    frequency_penalty: float | NotGiven = NOT_GIVEN,
    n_results: int | NotGiven = NOT_GIVEN,
    presence_penalty: float | NotGiven = NOT_GIVEN,
    temperature: float | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    **kwargs,
):
    params = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n_results,
        user=user,
        **kwargs,
    )

    return client.beta.chat.completions.parse(**params)


async def aparse(
    client: AsyncOpenAI | AsyncAzureOpenAI,
    model_name: str,
    messages: Any,
    max_tokens: int,
    top_p: float | NotGiven = NOT_GIVEN,
    stop: str | NotGiven = NOT_GIVEN,
    frequency_penalty: float | NotGiven = NOT_GIVEN,
    n_results: int | NotGiven = NOT_GIVEN,
    presence_penalty: float | NotGiven = NOT_GIVEN,
    temperature: float | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    **kwargs,
):
    params = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n_results,
        user=user,
        **kwargs,
    )

    return await client.beta.chat.completions.parse(**params)


class OpenAIChatCoreModule(BaseChatModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        organization_id_env_name: str | None = None,
        api_type: str = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        max_tokens: int = 2048,
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
    ) -> None:
        assert api_type in ["openai", "azure"], "api_type must be 'openai' or 'azure'."
        if api_type == "azure":
            assert (
                api_version and endpoint_env_name and deployment_id_env_name
            ), "api_version, endpoint_env_name, and deployment_id_env_name must be specified for Azure API."

        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.organization_id_env_name = organization_id_env_name
        self.max_tokens = max_tokens
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
        if model_name not in _OLDER_MODEL_CONFIG.keys():
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

        else:
            # TODO : add logging message
            if seed:
                print(
                    f"seed is ignored because it's not supported for {model_name} (api_type:{api_type})"
                )
            if json_mode:
                print(
                    f"response_format is ignored because it's not supported for {model_name} (api_type:{api_type})"
                )

        if system_instruction:
            system_instruction = OpenAIMessage.to_universal_message(
                role="system", message=system_instruction
            )
            self.system_instruction = OpenAIMessage.to_client_message(system_instruction)
        else:
            self.system_instruction = None

        self.conversation_length_adjuster = conversation_length_adjuster

    def run(
        self, messages: list[dict[str, str]], n_results: int | NotGiven = NOT_GIVEN
    ) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        client = get_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        if isinstance(self.response_format, ModelMetaclass):
            response = parse(
                client=client,
                model_name=self.model_name,
                messages=_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=NOT_GIVEN,
                n_results=n_results,
                user=self.user,
                **self.additional_inputs,
            )
        else:
            response = completion(
                client=client,
                model_name=self.model_name,
                messages=_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=NOT_GIVEN,
                stream=False,
                n_results=n_results,
                user=self.user,
                **self.additional_inputs,
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

        client = get_async_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        if isinstance(self.response_format, ModelMetaclass):
            response = await aparse(
                client=client,
                model_name=self.model_name,
                messages=_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=NOT_GIVEN,
                n_results=n_results,
                user=self.user,
                **self.additional_inputs,
            )
        else:
            response = await acompletion(
                client=client,
                model_name=self.model_name,
                messages=_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=NOT_GIVEN,
                stream=False,
                n_results=n_results,
                user=self.user,
                **self.additional_inputs,
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

    def stream(self, messages: list[dict[str, str]]) -> Generator[CompletionResults, None, None]:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        client = get_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        if self.api_type == "openai":
            stream_options = {"include_usage": True}
            additional_inputs = self.additional_inputs | {"stream_options": stream_options}
        else:
            # Azure OpenAI does not support stream_options
            additional_inputs = self.additional_inputs

        response = completion(
            client=client,
            model_name=self.model_name,
            messages=_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=NOT_GIVEN,
            stream=True,
            n_results=1,
            user=self.user,
            **additional_inputs,
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
                        if self.api_type == "azure":
                            # Azure OpenAI does not return prompt_tokens and completion_tokens
                            prompt_tokens = sum(
                                [get_n_tokens(m, self.model_name)["total"] for m in messages]
                            )

                            completion_tokens = get_n_tokens(
                                ChatCompletionAssistantMessageParam(
                                    role="assistant",
                                    content=[{"type": "text", "text": all_chunk}],
                                ),
                                self.model_name,
                            )["total"]

            else:
                if self.api_type == "openai":
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
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[CompletionResults, None]:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        client = get_async_client(
            api_key_env_name=self.api_key_env_name,
            organization_id_env_name=self.organization_id_env_name,
            api_version=self.api_version,
            endpoint_env_name=self.endpoint_env_name,
            deployment_id_env_name=self.deployment_id_env_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        _messages = [self.system_instruction] + messages if self.system_instruction else messages
        if self.conversation_length_adjuster:
            _messages = self.conversation_length_adjuster.run(_messages)

        if self.api_type == "openai":
            stream_options = {"include_usage": True}
            additional_inputs = self.additional_inputs | {"stream_options": stream_options}
        else:
            # Azure OpenAI does not support stream_options
            additional_inputs = self.additional_inputs

        response = await acompletion(
            client=client,
            model_name=self.model_name,
            messages=_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=NOT_GIVEN,
            stream=True,
            n_results=1,
            user=self.user,
            **additional_inputs,
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
                        if self.api_type == "azure":
                            # Azure OpenAI does not return prompt_tokens and completion_tokens
                            prompt_tokens = sum(
                                [get_n_tokens(m, self.model_name)["total"] for m in messages]
                            )

                            completion_tokens = get_n_tokens(
                                ChatCompletionAssistantMessageParam(
                                    role="assistant",
                                    content=[{"type": "text", "text": all_chunk}],
                                ),
                                self.model_name,
                            )["total"]

            else:
                if self.api_type == "openai":
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
        context_length: int | None = None,
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
        if model_name in MODEL_POINT.keys():
            print(f"{model_name} is automatically converted to {MODEL_POINT[model_name]}")
            model_name = MODEL_POINT[model_name]

        assert (
            model_name in MODEL_CONFIG.keys()
        ), f"model_name must be one of {', '.join(sorted(MODEL_CONFIG.keys()))}."

        token_lim = get_token_limit(model_name)
        max_tokens = max_tokens if max_tokens else MODEL_CONFIG[model_name]["max_output_tokens"]
        context_length = token_lim - max_tokens if context_length is None else context_length
        assert (
            token_lim >= max_tokens + context_length
        ), f"max_tokens({max_tokens}) + context_length({context_length}) must be less than or equal to the token limit of the model ({token_lim})."
        assert context_length > 0, "context_length must be positive."

        conversation_length_adjuster = (
            OldConversationTruncationModule(model_name=model_name, context_length=context_length)
            if conversation_length_adjuster is None
            else conversation_length_adjuster
        )

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
