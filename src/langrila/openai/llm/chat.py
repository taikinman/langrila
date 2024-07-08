import asyncio
from copy import deepcopy
from typing import Any, AsyncGenerator, Generator, Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from PIL import Image

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
from ...utils import make_batch
from ..conversation_adjuster.truncate import OldConversationTruncationModule
from ..message import OpenAIMessage
from ..model_config import _OLDER_MODEL_CONFIG, _VISION_MODEL, MODEL_CONFIG, MODEL_POINT
from ..openai_utils import get_async_client, get_client, get_n_tokens, get_token_limit


def completion(
    client: OpenAI | AzureOpenAI,
    model_name: str,
    messages: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stream: bool,
    stop: Optional[str] = None,
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
        **kwargs,
    )

    if model_name in _VISION_MODEL:
        params.pop("stop")

    return client.chat.completions.create(**params)


async def acompletion(
    client: AsyncOpenAI | AsyncAzureOpenAI,
    model_name: str,
    messages: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: bool,
    stream: bool,
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
        **kwargs,
    )

    if model_name in _VISION_MODEL:
        params.pop("stop")

    return await client.chat.completions.create(**params)


class OpenAIChatCoreModule(BaseChatModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        organization_id_env_name: Optional[str] = None,
        api_type: str = "openai",
        api_version: Optional[str] = None,
        endpoint_env_name: Optional[str] = None,
        deployment_id_env_name: Optional[str] = None,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        seed: Optional[int] = None,
        response_format: Optional[dict[str, str]] = None,
        system_instruction: str | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
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

        self.additional_inputs = {}
        if model_name not in _OLDER_MODEL_CONFIG.keys():
            self.seed = seed
            self.response_format = response_format
            self.additional_inputs["seed"] = seed

            if model_name not in _VISION_MODEL:
                self.additional_inputs["response_format"] = response_format
        else:
            # TODO : add logging message
            if seed:
                print(
                    f"seed is ignored because it's not supported for {model_name} (api_type:{api_type})"
                )
            if response_format:
                print(
                    f"response_format is ignored because it's not supported for {model_name} (api_type:{api_type})"
                )

        self.system_instruction = (
            OpenAIMessage(content=system_instruction).as_system if system_instruction else None
        )
        self.conversation_length_adjuster = conversation_length_adjuster

    def run(self, messages: list[dict[str, str]]) -> CompletionResults:
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

        response = completion(
            client=client,
            model_name=self.model_name,
            messages=_messages,
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
            **self.additional_inputs,
        )

        usage = Usage(model_name=self.model_name)
        usage += response.usage
        response_message = response.choices[0].message.content.strip("\n")
        return CompletionResults(
            usage=usage,
            message={"role": "assistant", "content": response_message},
            prompt=deepcopy(messages),
        )

    async def arun(self, messages: list[dict[str, str]]) -> CompletionResults:
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

        response = await acompletion(
            client=client,
            model_name=self.model_name,
            messages=_messages,
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
            **self.additional_inputs,
        )

        usage = Usage(model_name=self.model_name)
        usage += response.usage
        response_message = response.choices[0].message.content.strip("\n")
        return CompletionResults(
            usage=usage,
            message={"role": "assistant", "content": response_message},
            prompt=deepcopy(messages),
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

        response = completion(
            client=client,
            model_name=self.model_name,
            messages=_messages,
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=True,
            **self.additional_inputs,
        )

        response_message = {"role": "assistant", "content": ""}
        for r in response:
            if len(r.choices) > 0:
                delta = r.choices[0].delta
                if delta is not None:
                    chunk = delta.content
                    if chunk is not None:
                        response_message["content"] += chunk
                        yield CompletionResults(
                            usage=Usage(model_name=self.model_name),
                            message=response_message,
                            prompt=[{}],
                        )
                    else:
                        # at the end of stream, return the whole message and usage
                        usage = Usage(
                            model_name=self.model_name,
                            prompt_tokens=sum(
                                [get_n_tokens(m, self.model_name)["total"] for m in messages]
                            ),
                            completion_tokens=get_n_tokens(response_message, self.model_name)[
                                "total"
                            ],
                        )

                        yield CompletionResults(
                            usage=usage, message=response_message, prompt=deepcopy(messages)
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

        response = await acompletion(
            client=client,
            model_name=self.model_name,
            messages=_messages,
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=True,
            **self.additional_inputs,
        )

        response_message = {"role": "assistant", "content": ""}
        async for r in response:
            if len(r.choices) > 0:
                delta = r.choices[0].delta
                if delta is not None:
                    chunk = delta.content
                    if chunk is not None:
                        response_message["content"] += chunk
                        yield CompletionResults(
                            usage=Usage(model_name=self.model_name),
                            message=response_message,
                            prompt=[{}],
                        )
                    else:
                        # at the end of stream, return the whole message and usage
                        usage = Usage(
                            model_name=self.model_name,
                            prompt_tokens=sum(
                                [get_n_tokens(m, self.model_name)["total"] for m in messages]
                            ),
                            completion_tokens=get_n_tokens(response_message, self.model_name)[
                                "total"
                            ],
                        )

                        yield CompletionResults(
                            usage=usage, message=response_message, prompt=deepcopy(messages)
                        )


class OpenAIChatModule(ChatWrapperModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        api_type: str = "openai",
        api_version: Optional[str] = None,
        endpoint_env_name: Optional[str] = None,
        deployment_id_env_name: Optional[str] = None,
        organization_id_env_name: Optional[str] = None,
        max_tokens: Optional[int] = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        seed: Optional[int] = None,
        response_format: Optional[dict[str, str]] = None,
        context_length: Optional[int] = None,
        conversation_memory: Optional[BaseConversationMemory] = None,
        content_filter: Optional[BaseFilter] = None,
        conversation_length_adjuster: Optional[BaseConversationLengthAdjuster] = None,
        system_instruction: str | None = None,
        token_counter: TokenCounter | None = None,
    ):
        if model_name in MODEL_POINT.keys():
            print(f"{model_name} is automatically converted to {MODEL_POINT[model_name]}")
            model_name = MODEL_POINT[model_name]

        assert (
            model_name in MODEL_CONFIG.keys()
        ), f"model_name must be one of {', '.join(sorted(MODEL_CONFIG.keys()))}."

        token_lim = get_token_limit(model_name)
        max_tokens = max_tokens if max_tokens else int(token_lim / 2)
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
            response_format=response_format,
            system_instruction=system_instruction,
            conversation_length_adjuster=conversation_length_adjuster,
        )

        super().__init__(
            chat_model=chat_model,
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
        )

    def _get_client_message_type(self) -> type[BaseMessage]:
        return OpenAIMessage

    def run(
        self,
        prompt: str,
        images: Image.Image | bytes | list[Image.Image | bytes] | None = None,
        init_conversation: list[dict[str, Any]] | None = None,
        image_resolution: str = "low",
    ) -> CompletionResults:
        return super().run(
            prompt=prompt,
            images=images,
            init_conversation=init_conversation,
            image_resolution=image_resolution,
        )

    async def arun(
        self,
        prompt: str,
        images: Image.Image | bytes | list[Image.Image | bytes] | None = None,
        init_conversation: list[dict[str, Any]] | None = None,
        image_resolution: str | list[str] = "low",
    ) -> CompletionResults:
        return await super().arun(
            prompt=prompt,
            images=images,
            init_conversation=init_conversation,
            image_resolution=image_resolution,
        )

    def stream(
        self,
        prompt: str,
        images: Image.Image | bytes | list[Image.Image | bytes] | None = None,
        init_conversation: list[dict[str, Any]] | None = None,
        image_resolution: str = "low",
    ) -> Generator[CompletionResults, None, None]:
        return super().stream(
            prompt=prompt,
            images=images,
            init_conversation=init_conversation,
            image_resolution=image_resolution,
        )

    async def astream(
        self,
        prompt: str,
        images: Image.Image | bytes | list[Image.Image | bytes] | None = None,
        init_conversation: list[dict[str, Any]] | None = None,
        image_resolution: str | list[str] = "low",
    ) -> AsyncGenerator[CompletionResults, None]:
        return await super().astream(
            prompt=prompt,
            images=images,
            init_conversation=init_conversation,
            image_resolution=image_resolution,
        )

    async def abatch_run(
        self,
        prompts: list[str],
        images: list[Image.Image | bytes | list[Image.Image, bytes]] | None = None,
        init_conversations: Optional[list[list[dict[str, str]]]] = None,
        image_resolutions: str | list[str] = "low",
        batch_size: int = 4,
    ) -> list[CompletionResults]:
        if init_conversations is None:
            init_conversations = [None] * len(prompts)

        if images is None:
            images = [None] * len(prompts)

        if isinstance(image_resolutions, str):
            image_resolutions = [image_resolutions] * len(prompts)

        z = zip(prompts, init_conversations, images, image_resolutions, strict=True)
        batches = make_batch(list(z), batch_size)
        results = []
        for batch in batches:
            async_processes = [
                self.arun(
                    prompt=prompt,
                    images=_images,
                    init_conversation=init_conversation,
                    image_resolution=resolution,
                )
                for prompt, init_conversation, _images, resolution in batch
            ]
            results.extend(await asyncio.gather(*async_processes))
        return results
