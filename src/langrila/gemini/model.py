import copy
from typing import Any, AsyncGenerator, Callable, Generator, Literal, Optional, Sequence

from google.auth import credentials as auth_credentials

from ..base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseMessage,
)
from ..llm_wrapper import ChatWrapperModule
from ..message_content import ConversationType, InputType
from ..result import CompletionResults, FunctionCallingResults
from ..usage import TokenCounter, Usage
from .gemini_utils import get_message_cls, get_model
from .llm.chat import GeminiChatModule
from .llm.function_calling import GeminiFunctionCallingModule


class Gemini:
    def __init__(
        self,
        model_name: str,
        api_key_env_name: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: int = 60,
        content_filter: BaseFilter | None = None,
        conversation_memory: BaseConversationMemory | None = None,
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
        tools: list[Callable] | None = None,
        tool_configs: list[Any] | None = None,
    ):
        self.chat = GeminiChatModule(
            model_name=model_name,
            api_key_env_name=api_key_env_name,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
            content_filter=content_filter,
            conversation_memory=conversation_memory,
            system_instruction=system_instruction,
            token_counter=token_counter,
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

        self.function_calling = GeminiFunctionCallingModule(
            model_name=model_name,
            api_key_env_name=api_key_env_name,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
            content_filter=content_filter,
            conversation_memory=conversation_memory,
            system_instruction=system_instruction,
            token_counter=token_counter,
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
            tools=tools,
            tool_configs=tool_configs,
        )

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
        tool_only: bool = False,
    ) -> CompletionResults | FunctionCallingResults:
        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        if tool_choice is not None:
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if tool_only:
                return response_function_calling

            prompt = [c for r in response_function_calling.results for c in r.content]

        response_chat: CompletionResults = self.chat.run(
            prompt, init_conversation=init_conversation
        )

        return response_chat

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
        tool_only: bool = False,
    ) -> CompletionResults | FunctionCallingResults:
        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        if tool_choice is not None:
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if tool_only:
                return response_function_calling

            prompt = [c for r in response_function_calling.results for c in r.content]

        response_chat: CompletionResults = await self.chat.arun(
            prompt, init_conversation=init_conversation
        )

        return response_chat

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
    ) -> Generator[CompletionResults, None, None]:
        if tool_choice is not None:
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            prompt = [c for r in response_function_calling.results for c in r.content]

        response_chat: CompletionResults = self.chat.stream(
            prompt, init_conversation=init_conversation
        )

        return response_chat

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
    ) -> AsyncGenerator[CompletionResults, None]:
        if tool_choice is not None:
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            prompt = [c for r in response_function_calling.results for c in r.content]

        response_chat: CompletionResults = self.chat.astream(
            prompt, init_conversation=init_conversation
        )

        return response_chat
