import json
from logging import Logger
from typing import Any, AsyncGenerator, Callable, Generator, Generic, Sequence, cast

from pydantic_core import ValidationError

from .client import LLMClient
from .config import AgentConfig
from .embedding import EmbeddingResults
from .logger import DEFAULT_LOGGER as default_logger
from .memory import BaseConversationMemory
from .message import Message
from .model import LLMModel
from .prompt import Prompt, PromptType, TextPrompt, ToolUsePrompt
from .pydantic import BaseModel
from .response import Response, ResponseType, TextResponse, ToolCallResponse
from .tool import Tool
from .typing import ClientMessage, ClientMessageContent, ClientTool
from .usage import Usage

AgentInput = (
    Prompt
    | PromptType
    | Response
    | ResponseType
    | list[Prompt | PromptType | ResponseType | Response]
)


def format_validation_error_msg(e: ValidationError, tool_name: str) -> str:
    error_txt = ""
    for error in e.errors():
        error_txt += (
            f"Tool: {tool_name}\nAttribute: {error['loc'][0]}\n"
            f"Input value: {error['input']}\n{e.__class__.__name__}: {error['msg']}\n\n"
        )
    return error_txt.strip()


class Agent(Generic[ClientMessage, ClientMessageContent, ClientTool]):
    def __init__(
        self,
        client: LLMClient[ClientMessage, ClientMessageContent, ClientTool] | None = None,
        llm: LLMModel[ClientMessage, ClientMessageContent, ClientTool] | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        agent_config: AgentConfig | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        logger: Logger | None = None,
        response_schema: BaseModel | None = None,
        **kwargs: Any,
    ):
        self._client = client
        self.agent_config = agent_config or AgentConfig()
        self.retry_prompt = self.agent_config.retry_prompt
        self.conversation_memory = conversation_memory
        self.n_validation_retries = self.agent_config.n_validation_retries
        self.logger = logger or default_logger
        self.response_schema = response_schema
        self.__response_schema_name = "final_answer"
        self.__final_response_prompt = "Final answer please."
        self.__max_repeat_text_response = 3
        _tools = tools or []

        if response_schema:
            _tools += self._prepare_tool_as_response_schema(response_schema)

        if llm is not None:
            self.llm = llm

            if conversation_memory is not None and self.llm.conversation_memory is not None:
                # If both conversation_memory and llm.conversation_memory are provided,
                # then llm.conversation_memory will be ignored.
                self.llm.conversation_memory = None
        else:
            assert client is not None, "Either client or llm must be provided"

            self.llm = LLMModel(
                client=client,
                logger=logger,
                tools=_tools,
                **kwargs,
            )

        if tools is not None:
            self._tools = {tool.name: tool for tool in self.llm._prepare_tools(_tools)}

    def _prepare_tool_as_response_schema(self, response_schema: BaseModel) -> list[Tool]:
        _response_schema_json = response_schema.model_json_schema()
        return self._prepare_response_schema_tool(_response_schema_json)

    def _update_usage(self, usage: Usage, response: Response) -> Usage:
        return cast(
            Usage,
            usage.update(
                **{
                    "prompt_tokens": usage.prompt_tokens + (response.usage.prompt_tokens or 0),
                    "output_tokens": usage.output_tokens + (response.usage.output_tokens or 0),
                }
            ),
        )

    def _is_any_text_response(self, response: Response) -> bool:
        if response.contents is None:
            return False
        return any([isinstance(content, TextResponse) for content in response.contents])

    def _is_all_text_response(self, response: Response) -> bool:
        if response.contents is None:
            return False
        return all([isinstance(content, TextResponse) for content in response.contents])

    def _include_tool_call_response(self, response: Response) -> bool:
        if response.contents is None:
            return False
        return any([isinstance(content, ToolCallResponse) for content in response.contents])

    def generate_text(self, prompt: AgentInput, **kwargs: Any) -> Response:
        messages = self.load_history()

        messages.extend(self._process_user_prompt(prompt))

        n_validation_retries = 0
        n_repeat_text_response = 0
        final_result = None
        total_usage = Usage()
        while (
            n_validation_retries < self.n_validation_retries
            and n_repeat_text_response < self.__max_repeat_text_response
        ):
            response = self.llm.generate_text(messages=messages, **kwargs)
            total_usage = self._update_usage(total_usage, response)

            if not response.contents:
                self.logger.error("No response received from the model. Retrying.")
                continue

            messages.append(response)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(response)
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                messages.append(final_result)
                break
            else:
                if is_error:
                    n_validation_retries += 1
                else:
                    n_validation_retries = 0

                    if self.response_schema is None:
                        if self._include_tool_call_response(response):
                            continue
                        else:
                            break
                    else:
                        if self._is_all_text_response(response):
                            n_repeat_text_response += 1
                            messages.append(
                                Prompt(contents=self.__final_response_prompt, role="user")
                            )
                        else:
                            n_repeat_text_response = 0

        self._tmp = messages
        self.store_history(messages)

        if final_result:
            return final_result

        response.usage = total_usage
        return response

    async def generate_text_async(self, prompt: AgentInput, **kwargs: Any) -> Response:
        messages = self.load_history()

        messages.extend(self._process_user_prompt(prompt))

        n_validation_retries = 0
        n_repeat_text_response = 0
        final_result = None
        total_usage = Usage()
        while (
            n_validation_retries < self.n_validation_retries
            and n_repeat_text_response < self.__max_repeat_text_response
        ):
            response = await self.llm.generate_text_async(messages=messages, **kwargs)
            total_usage = self._update_usage(total_usage, response)

            if not response.contents:
                self.logger.error("No response received from the model. Retrying.")
                continue

            messages.append(response)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(response)
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                messages.append(final_result)
                break
            else:
                if is_error:
                    n_validation_retries += 1
                else:
                    n_validation_retries = 0

                    if self.response_schema is None:
                        if self._include_tool_call_response(response):
                            continue
                        else:
                            break
                    else:
                        if self._is_all_text_response(response):
                            n_repeat_text_response += 1
                            messages.append(
                                Prompt(contents=self.__final_response_prompt, role="user")
                            )
                        else:
                            n_repeat_text_response = 0

        self.store_history(messages)

        if final_result:
            return final_result

        response.usage = total_usage
        return response

    def stream_text(self, prompt: AgentInput, **kwargs: Any) -> Generator[Response, None, None]:
        messages = self.load_history()

        messages.extend(self._process_user_prompt(prompt))

        n_validation_retries = 0
        n_repeat_text_response = 0
        final_result = None
        total_usage = Usage()
        while (
            n_validation_retries < self.n_validation_retries
            and n_repeat_text_response < self.__max_repeat_text_response
        ):
            streamed_response = self.llm.stream_text(messages=messages, **kwargs)
            for chunk in streamed_response:
                if not chunk.is_last_chunk:
                    yield chunk

            messages.append(chunk)

            total_usage = self._update_usage(total_usage, chunk)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(chunk)
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                messages.append(final_result)
                self.store_history(messages)
                final_result.usage = total_usage
                yield final_result
                break
            else:
                if is_error:
                    n_validation_retries += 1
                else:
                    n_validation_retries = 0

                    total_usage = self._update_usage(total_usage, chunk)
                    if self.response_schema is None:
                        # chunk.usage = total_usage
                        # messages.append(chunk)
                        self.store_history(messages)
                        # yield chunk

                        if self._include_tool_call_response(chunk):
                            continue
                        else:
                            break
                    else:
                        if self._is_all_text_response(chunk):
                            n_repeat_text_response += 1
                            messages.append(
                                Prompt(contents=self.__final_response_prompt, role="user")
                            )
                        else:
                            n_repeat_text_response = 0

    async def stream_text_async(
        self, prompt: AgentInput, **kwargs: Any
    ) -> AsyncGenerator[Response, None]:
        messages = self.load_history()

        messages.extend(self._process_user_prompt(prompt))

        n_validation_retries = 0
        n_repeat_text_response = 0
        final_result = None
        total_usage = Usage()
        while (
            n_validation_retries < self.n_validation_retries
            and n_repeat_text_response < self.__max_repeat_text_response
        ):
            streamed_response = self.llm.stream_text_async(messages=messages, **kwargs)
            async for chunk in streamed_response:
                if not chunk.is_last_chunk:
                    yield chunk

            messages.append(chunk)

            total_usage = self._update_usage(total_usage, chunk)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(chunk)
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                messages.append(final_result)
                self.store_history(messages)
                final_result.usage = total_usage
                yield final_result
                break
            else:
                if is_error:
                    n_validation_retries += 1
                else:
                    n_validation_retries = 0

                    total_usage = self._update_usage(total_usage, chunk)
                    if self.response_schema is None:
                        # chunk.usage = total_usage
                        # messages.append(chunk)
                        self.store_history(messages)
                        # yield chunk

                        if self._include_tool_call_response(chunk):
                            continue
                        else:
                            break
                    else:
                        if self._is_all_text_response(chunk):
                            n_repeat_text_response += 1
                            messages.append(
                                Prompt(contents=self.__final_response_prompt, role="user")
                            )
                        else:
                            n_repeat_text_response = 0

    def _process_user_prompt(self, prompt: AgentInput) -> list[Prompt | Response]:
        if isinstance(prompt, (Prompt, Response)):
            return [prompt]
        elif isinstance(prompt, (str, PromptType)):
            return [Prompt(role="user", contents=prompt)]
        elif isinstance(prompt, ResponseType):
            return [Response(role="assistant", contents=[prompt], usage=Usage())]
        elif isinstance(prompt, list):
            messages = []
            for p in prompt:
                messages.extend(self._process_user_prompt(p))
            return messages

    def _is_tool_call_response(self, content: ResponseType) -> bool:
        if isinstance(content, ToolCallResponse):
            return True
        return False

    def _is_text_response(self, content: ResponseType) -> bool:
        if isinstance(content, TextResponse):
            return True
        return False

    def _validate_tools_and_prepare_next_message(
        self, response: Response
    ) -> tuple[Prompt | None, bool, Response | None]:
        next_turn_contents: list[PromptType] = []
        is_error = False
        final_result = None
        for content in response.contents or []:
            if isinstance(content, ToolCallResponse):
                tool_name = content.name
                try:
                    tool = self._tools[tool_name]
                except KeyError:
                    self.logger.error(f"Tool: {tool_name} is not found in the agent. Retrying...")
                    next_turn_contents.append(
                        ToolUsePrompt(
                            error=f"Tool: {tool_name} is not found in the agent. Please use other tools.",
                            call_id=content.call_id,
                            args=content.args,
                            name=tool_name,
                        )
                    )
                    continue

                schema_validator = tool.schema_validator
                context = tool.context or {}

                if not content.args:
                    continue

                try:
                    args = json.loads(content.args)
                    self.logger.info(f"Running tool: {tool_name}")

                    if tool_name != self.__response_schema_name:
                        assert (
                            schema_validator is not None
                        ), f"Schema validator is required for tool: {tool_name} but it's NoneType object. "
                        args = schema_validator.validate_python({**args, **context})
                        tool_result = tool.run(args)
                        next_turn_contents.append(
                            ToolUsePrompt(
                                output=str(tool_result),
                                call_id=content.call_id,
                                args=content.args,
                                name=tool_name,
                            )
                        )

                    else:
                        assert (
                            self.response_schema is not None
                        ), "Please provide response_schema in the agent."
                        self.response_schema.model_validate(args)
                        next_turn_contents.append(
                            ToolUsePrompt(
                                output=json.dumps(args),
                                call_id=content.call_id,
                                args=content.args,
                                name=tool_name,
                            )
                        )

                        final_result = Response(
                            role="assistant",
                            contents=[TextResponse(text=json.dumps(args))],
                            usage=response.usage,
                            raw=response.raw,
                            name=response.name,
                        )

                    self.logger.info(f"Tool: {tool_name} successfully ran.")

                except ValidationError as e:
                    self.logger.error(
                        f"Validation error occurred while running tool. Retrying...\n"
                        f"{format_validation_error_msg(e, tool_name)}"
                    )

                    next_turn_contents.append(
                        ToolUsePrompt(
                            error=format_validation_error_msg(e, tool_name),
                            call_id=content.call_id,
                            args=content.args,
                            name=tool_name,
                        )
                    )

                    is_error = True

                except Exception as e:
                    self.logger.error(
                        f"Error occurred while running. Retrying...\n"
                        f"Tool: {tool_name}\n{e.__class__.__name__}: {str(e)}"
                    )

                    next_turn_contents.append(
                        ToolUsePrompt(
                            error=f"Tool: {tool_name}\nError: {str(e)}",
                            call_id=content.call_id,
                            args=content.args,
                            name=tool_name,
                        )
                    )

                    is_error = True

        if is_error:
            next_turn_contents.append(TextPrompt(text=self.retry_prompt.validation))

        if len(next_turn_contents) != 0:
            next_turn_message = Prompt(contents=next_turn_contents, role="user")
        else:
            next_turn_message = None

        return next_turn_message, is_error, final_result

    def store_history(self, messages: list[Prompt | Response]) -> None:
        if self.conversation_memory is not None:
            self.conversation_memory.store(
                [
                    m.model_dump(
                        include={"role", "contents", "name", "usage", "type"}, exclude={"raw"}
                    )
                    for m in messages
                    if m.contents
                ]
            )

    def load_history(self) -> list[Prompt | Response]:
        if self.conversation_memory is not None:
            messages = self.conversation_memory.load()
            return [
                Message[eval(m["type"])].model_validate({"message": m}).message  # type: ignore[misc]
                for m in messages
            ]
        return []

    def _prepare_response_schema_tool(self, response_schema: dict[str, Any]) -> list[Tool]:
        return [
            Tool(
                name=self.__response_schema_name,
                description="The final answer which ends this conversation. Must run at the end of the conversation.",
                schema_dict=response_schema,
            )
        ]

    def generate_image(self, prompt: str, **kwargs: Any) -> Response:
        return self.llm.generate_image(prompt, **kwargs)

    async def generate_image_async(self, prompt: str, **kwargs: Any) -> Response:
        return await self.llm.generate_image_async(prompt, **kwargs)

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        return self.llm.embed_text(texts, **kwargs)

    async def embed_text_async(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        return await self.llm.embed_text_async(texts, **kwargs)
