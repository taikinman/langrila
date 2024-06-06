import copy
from typing import AsyncGenerator, Generator, Optional

from google.generativeai.types.generation_types import GenerationConfig
from google.generativeai.types.helper_types import RequestOptions

from ...base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseMessage,
)
from ...llm_wrapper import ChatWrapperModule
from ...result import CompletionResults
from ...usage import Usage
from ..message import GeminiMessage
from ..utils import get_model


class GeminiChatCoreModule(BaseChatModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: int = 60,
    ):
        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.max_output_tokens = max_tokens

        self.generation_config = GenerationConfig(
            stop_sequences=None,
            max_output_tokens=self.max_output_tokens,
            temperature=0.0,
            top_p=0.0,
            response_mime_type="text/plain" if not json_mode else "application/json",
        )

        self.request_options = RequestOptions(
            timeout=timeout,
        )

    def run(self, messages: list[dict[str, str]]) -> CompletionResults:
        model = get_model(self.model_name, self.api_key_env_name)
        response = model.generate_content(contents=messages, request_options=self.request_options)
        content = response.candidates[0].content
        return CompletionResults(
            message={"role": content.role, "parts": [content.parts[0].text]},
            usage=Usage(
                prompt_tokens=model.count_tokens(messages).total_tokens,
                completion_tokens=model.count_tokens(content.parts).total_tokens,
            ),
            prompt=copy.deepcopy(messages),
        )

    async def arun(self, messages: list[dict[str, str]]) -> CompletionResults:
        model = get_model(self.model_name, self.api_key_env_name)
        response = await model.generate_content_async(
            contents=messages, request_options=self.request_options
        )
        content = response.candidates[0].content
        return CompletionResults(
            message={"role": content.role, "parts": [content.parts[0].text]},
            usage=Usage(
                prompt_tokens=(await model.count_tokens_async(messages)).total_tokens,
                completion_tokens=(await model.count_tokens_async(content.parts)).total_tokens,
            ),
            prompt=copy.deepcopy(messages),
        )

    def stream(
        self, messages: list[dict[str, str | list[str]]]
    ) -> Generator[CompletionResults, None, None]:
        model = get_model(self.model_name, self.api_key_env_name)
        responses = model.generate_content(
            contents=messages, request_options=self.request_options, stream=True
        )

        entire_response_texts = []
        for response in responses:
            content = response.candidates[0].content
            entire_response_texts.extend([content.parts[0].text])
            result = CompletionResults(
                message={"role": content.role, "parts": ["".join(entire_response_texts)]},
                usage=Usage(),
                prompt="",
            )

            yield result

        # at the end of the stream, return the entire response
        entire_response_texts = "".join(entire_response_texts)
        yield CompletionResults(
            message={"role": content.role, "parts": [entire_response_texts]},
            usage=Usage(
                prompt_tokens=model.count_tokens(messages).total_tokens,
                completion_tokens=model.count_tokens(entire_response_texts).total_tokens,
            ),
            prompt=messages,
        )

    async def astream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[CompletionResults, None]:
        model = get_model(self.model_name, self.api_key_env_name)
        responses = await model.generate_content_async(
            contents=messages, request_options=self.request_options, stream=True
        )

        entire_response_texts = []
        async for response in responses:
            content = response.candidates[0].content
            entire_response_texts.extend([content.parts[0].text])
            result = CompletionResults(
                message={"role": content.role, "parts": ["".join(entire_response_texts)]},
                usage=Usage(),
                prompt="",
            )

            yield result

        # at the end of the stream, return the entire response
        entire_response_texts = "".join(entire_response_texts)
        yield CompletionResults(
            message={"role": content.role, "parts": [entire_response_texts]},
            usage=Usage(
                prompt_tokens=(await model.count_tokens_async(messages)).total_tokens,
                completion_tokens=(
                    await model.count_tokens_async(entire_response_texts)
                ).total_tokens,
            ),
            prompt=messages,
        )


class GeminiChatModule(ChatWrapperModule):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: int = 60,
        content_filter: Optional[BaseFilter] = None,
        conversation_memory: Optional[BaseConversationMemory] = None,
        conversation_length_adjuster: Optional[BaseConversationLengthAdjuster] = None,
    ):
        chat_model = GeminiChatCoreModule(
            api_key_env_name=api_key_env_name,
            model_name=model_name,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
        )

        super().__init__(
            chat_model=chat_model,
            content_filter=content_filter,
            conversation_memory=conversation_memory,
            conversation_length_adjuster=conversation_length_adjuster,
        )

    def _get_client_message_type(self) -> type[BaseMessage]:
        return GeminiMessage
