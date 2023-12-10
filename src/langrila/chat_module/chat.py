import asyncio
from typing import Optional

from ..base import BaseConversationLengthAdjuster, BaseConversationMemory, BaseFilter, BaseModule
from ..conversation_adjuster.truncate import OldConversationTruncationModule
from ..message import Message
from ..model_config import _NEWER_MODEL_CONFIG, MODEL_CONFIG, MODEL_POINT
from ..result import CompletionResults
from ..usage import Usage
from ..utils import get_async_client, get_client, get_n_tokens, get_token_limit, make_batch


class BaseChatModule(BaseModule):
    def __init__(
        self,
        api_key_name: str,
        model_name: str,
        organization_id_name: Optional[str] = None,
        api_type: str = "openai",
        api_version: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        deployment_id_name: Optional[str] = None,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        seed: Optional[int] = None,
        response_format: Optional[dict[str, str]] = None,
    ) -> None:
        assert api_type in ["openai", "azure"], "api_type must be 'openai' or 'azure'."
        if api_type == "azure":
            assert (
                api_version and endpoint_name and deployment_id_name
            ), "api_version, endpoint_name, and deployment_id_name must be specified for Azure API."

        self.api_key_name = api_key_name
        self.model_name = model_name
        self.organization_id_name = organization_id_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_type = api_type
        self.api_version = api_version
        self.endpoint_name = endpoint_name
        self.deployment_id_name = deployment_id_name

        self.additional_inputs = {}
        if api_type == "openai" and model_name in _NEWER_MODEL_CONFIG.keys():
            self.seed = seed
            self.response_format = response_format
            self.additional_inputs["seed"] = seed
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

    def run(self, messages: list[dict[str, str]]) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        client = get_client(
            api_key_name=self.api_key_name,
            organization_id_name=self.organization_id_name,
            api_version=self.api_version,
            endpoint_name=self.endpoint_name,
            deployment_id_name=self.deployment_id_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            **self.additional_inputs,
        )

        usage = Usage()
        usage += response.usage
        response_message = response.choices[0].message.content.strip("\n")
        return CompletionResults(
            usage=usage,
            message={"role": "assistant", "content": response_message},
            prompt=messages
        )

    async def arun(self, messages: list[dict[str, str]]) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        client = get_async_client(
            api_key_name=self.api_key_name,
            organization_id_name=self.organization_id_name,
            api_version=self.api_version,
            endpoint_name=self.endpoint_name,
            deployment_id_name=self.deployment_id_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            **self.additional_inputs,
        )

        usage = Usage()
        usage += response.usage
        response_message = response.choices[0].message.content.strip("\n")
        return CompletionResults(
            usage=usage,
            message={"role": "assistant", "content": response_message},
            prompt=messages
        )

    def stream(self, messages: list[dict[str, str]]) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        client = get_client(
            api_key_name=self.api_key_name,
            organization_id_name=self.organization_id_name,
            api_version=self.api_version,
            endpoint_name=self.endpoint_name,
            deployment_id_name=self.deployment_id_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
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
                    if chunk is not None and chunk != "":
                        response_message["content"] += chunk
                        yield chunk

        usage = Usage(
            prompt_tokens=sum([get_n_tokens(m, self.model_name)["total"] for m in messages]),
            completion_tokens=get_n_tokens(response_message, self.model_name)["total"],
        )

        yield CompletionResults(
            usage=usage,
            message=response_message,
            prompt=messages
        )


    async def astream(self, messages: list[dict[str, str]]) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        if not isinstance(messages, list):
            raise ValueError("messages type must be list.")

        client = get_async_client(
            api_key_name=self.api_key_name,
            organization_id_name=self.organization_id_name,
            api_version=self.api_version,
            endpoint_name=self.endpoint_name,
            deployment_id_name=self.deployment_id_name,
            api_type=self.api_type,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
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
                    if chunk is not None and chunk != "":
                        response_message["content"] += chunk
                        yield chunk

        usage = Usage(
            prompt_tokens=sum([get_n_tokens(m, self.model_name)["total"] for m in messages]),
            completion_tokens=get_n_tokens(response_message, self.model_name)["total"],
        )

        yield CompletionResults(
            usage=usage,
            message=response_message,
            prompt=messages
        )


class OpenAIChatModule(BaseModule):
    def __init__(
        self,
        api_key_name: str,
        model_name: str,
        api_type: str = "openai",
        api_version: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        deployment_id_name: Optional[str] = None,
        organization_id_name: Optional[str] = None,
        max_tokens: Optional[int] = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        seed: Optional[int] = None,
        response_format: Optional[dict[str, str]] = None,
        context_length: Optional[int] = None,
        conversation_memory: Optional[BaseConversationMemory] = None,
        content_filter: Optional[BaseFilter] = None,
        conversation_length_adjuster: Optional[BaseConversationLengthAdjuster] = None,
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

        self.chat_model = BaseChatModule(
            api_key_name=api_key_name,
            organization_id_name=organization_id_name,
            model_name=model_name,
            max_tokens=max_tokens,
            api_type=api_type,
            api_version=api_version,
            endpoint_name=endpoint_name,
            deployment_id_name=deployment_id_name,
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
            response_format=response_format,
        )

        self.conversation_length_adjuster = (
            OldConversationTruncationModule(model_name=model_name, context_length=context_length)
            if conversation_length_adjuster is None
            else conversation_length_adjuster
        )
        self.conversation_memory = conversation_memory
        self.content_filter = content_filter

    def run(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
    ) -> CompletionResults:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt).as_user)

        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)

        messages = self.conversation_length_adjuster(messages)

        response = self.chat_model(messages)

        if self.content_filter is not None:
            response.message = self.content_filter.restore([response.message])[0]

        messages.append(response.message)

        if self.conversation_memory is not None:
            self.conversation_memory.store(messages)

        return response

    def stream(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
    ) -> CompletionResults:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt).as_user)

        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)

        messages = self.conversation_length_adjuster(messages)

        response = self.chat_model.stream(messages)

        response_message_stream = {"role": "assistant", "content": ""}
        for chunk in response:
            if isinstance(chunk, str):
                response_message_stream["content"] += chunk

                if self.content_filter is not None:
                    response_message_stream = self.content_filter.restore([response_message_stream])[0]

                yield response_message_stream["content"]
            elif isinstance(chunk, CompletionResults):
                if self.content_filter is not None:
                    chunk.message = self.content_filter.restore([chunk.message])[0]

                messages.append(chunk.message)

                if self.conversation_memory is not None:
                    self.conversation_memory.store(messages)

                yield chunk
            else:
                raise AssertionError

    async def astream(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
    ) -> CompletionResults:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt).as_user)

        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)

        messages = self.conversation_length_adjuster(messages)

        response = self.chat_model.astream(messages)

        response_message_stream = {"role": "assistant", "content": ""}
        async for chunk in response:
            if isinstance(chunk, str):
                response_message_stream["content"] += chunk

                if self.content_filter is not None:
                    response_message_stream = self.content_filter.restore([response_message_stream])[0]

                yield response_message_stream["content"]
            elif isinstance(chunk, CompletionResults):
                if self.content_filter is not None:
                    chunk.message = self.content_filter.restore([chunk.message])[0]

                messages.append(chunk.message)

                if self.conversation_memory is not None:
                    self.conversation_memory.store(messages)

                yield chunk
            else:
                raise AssertionError

    async def arun(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
    ) -> CompletionResults:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt).as_user)

        if self.content_filter is not None:
            messages = await self.content_filter(messages, arun=True)

        messages = self.conversation_length_adjuster(messages)

        response = await self.chat_model(messages, arun=True)

        if self.content_filter is not None:
            response.message = self.content_filter.restore([response.message])[0]

        messages.append(response.message)

        if self.conversation_memory is not None:
            self.conversation_memory.store(messages)

        return response

    async def abatch_run(
        self,
        prompts: list[str],
        init_conversations: Optional[list[list[dict[str, str]]]] = None,
        batch_size: int = 4,
    ) -> list[CompletionResults]:
        if init_conversations is None:
            init_conversations = [None] * len(prompts)

        assert (
            len(prompts) == len(init_conversations)
        ), "Length of prompts, init_conversations, and function_message_list must be the same."

        z = zip(prompts, init_conversations)
        batches = make_batch(list(z), batch_size)
        results = []
        for batch in batches:
            async_processes = [
                self.arun(prompt, init_conversation)
                for prompt, init_conversation in batch
            ]
            results.extend(await asyncio.gather(*async_processes))
        return results
