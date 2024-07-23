import copy
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
from ..gemini_utils import get_message_cls, get_model


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

        self.additional_kwargs = {}
        if api_type == "genai":
            from google.generativeai.types.helper_types import RequestOptions

            request_options = RequestOptions(
                timeout=timeout,
            )
            self.additional_kwargs["request_options"] = request_options

        self.system_instruction = system_instruction

    def run(self, messages: list[dict[str, str]]) -> CompletionResults:
        model = get_model(
            model_name=self.model_name,
            api_key_env_name=self.api_key_env_name,
            max_output_tokens=self.max_output_tokens,
            json_mode=self.json_mode,
            system_instruction=self.system_instruction,
            api_type=self.api_type,
            project_id_env_name=self.project_id_env_name,
            location_env_name=self.location_env_name,
            experiment=self.experiment,
            experiment_description=self.experiment_description,
            experiment_tensorboard=self.experiment_tensorboard,
            staging_bucket=self.staging_bucket,
            credentials=self.credentials,
            encryption_spec_key_name=self.encryption_spec_key_name,
            network=self.network,
            service_account=self.service_account,
            endpoint_env_name=self.endpoint_env_name,
            request_metadata=self.request_metadata,
            response_schema=self.response_schema,
        )
        response = model.generate_content(contents=messages, **self.additional_kwargs)
        content = response.candidates[0].content
        usage_metadata = response.usage_metadata
        return CompletionResults(
            message=content,
            usage=Usage(
                model_name=self.model_name,
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
            ),
            prompt=copy.deepcopy(messages),
        )

    async def arun(self, messages: list[dict[str, str]]) -> CompletionResults:
        model = get_model(
            model_name=self.model_name,
            api_key_env_name=self.api_key_env_name,
            max_output_tokens=self.max_output_tokens,
            json_mode=self.json_mode,
            system_instruction=self.system_instruction,
            api_type=self.api_type,
            project_id_env_name=self.project_id_env_name,
            location_env_name=self.location_env_name,
            experiment=self.experiment,
            experiment_description=self.experiment_description,
            experiment_tensorboard=self.experiment_tensorboard,
            staging_bucket=self.staging_bucket,
            credentials=self.credentials,
            encryption_spec_key_name=self.encryption_spec_key_name,
            network=self.network,
            service_account=self.service_account,
            endpoint_env_name=self.endpoint_env_name,
            request_metadata=self.request_metadata,
            response_schema=self.response_schema,
        )
        response = await model.generate_content_async(contents=messages, **self.additional_kwargs)
        content = response.candidates[0].content
        usage_metadata = response.usage_metadata
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
        model = get_model(
            model_name=self.model_name,
            api_key_env_name=self.api_key_env_name,
            max_output_tokens=self.max_output_tokens,
            json_mode=self.json_mode,
            system_instruction=self.system_instruction,
            api_type=self.api_type,
            project_id_env_name=self.project_id_env_name,
            location_env_name=self.location_env_name,
            experiment=self.experiment,
            experiment_description=self.experiment_description,
            experiment_tensorboard=self.experiment_tensorboard,
            staging_bucket=self.staging_bucket,
            credentials=self.credentials,
            encryption_spec_key_name=self.encryption_spec_key_name,
            network=self.network,
            service_account=self.service_account,
            endpoint_env_name=self.endpoint_env_name,
            request_metadata=self.request_metadata,
            response_schema=self.response_schema,
        )
        responses = model.generate_content(contents=messages, stream=True, **self.additional_kwargs)

        chunk_all = ""
        for response in responses:
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
                prompt_tokens=model.count_tokens(messages).total_tokens,
                completion_tokens=model.count_tokens(chunk_all).total_tokens,
            ),
            prompt=copy.deepcopy(messages),
        )

    async def astream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[CompletionResults, None]:
        model = get_model(
            model_name=self.model_name,
            api_key_env_name=self.api_key_env_name,
            max_output_tokens=self.max_output_tokens,
            json_mode=self.json_mode,
            system_instruction=self.system_instruction,
            api_type=self.api_type,
            project_id_env_name=self.project_id_env_name,
            location_env_name=self.location_env_name,
            experiment=self.experiment,
            experiment_description=self.experiment_description,
            experiment_tensorboard=self.experiment_tensorboard,
            staging_bucket=self.staging_bucket,
            credentials=self.credentials,
            encryption_spec_key_name=self.encryption_spec_key_name,
            network=self.network,
            service_account=self.service_account,
            endpoint_env_name=self.endpoint_env_name,
            request_metadata=self.request_metadata,
            response_schema=self.response_schema,
        )
        responses = await model.generate_content_async(
            contents=messages, stream=True, **self.additional_kwargs
        )

        chunk_all = ""
        async for _response in responses:
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
                prompt_tokens=(await model.count_tokens_async(messages)).total_tokens,
                completion_tokens=(await model.count_tokens_async(chunk_all)).total_tokens,
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
    ):
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
        )

        super().__init__(
            chat_model=chat_model,
            content_filter=content_filter,
            conversation_memory=conversation_memory,
            token_counter=token_counter,
        )

    def _get_client_message_type(self) -> type[BaseMessage]:
        return get_message_cls(self.chat_model.api_type)
