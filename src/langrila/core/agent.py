import json
import types
from logging import Logger
from typing import Any, AsyncGenerator, Callable, Generator, Generic, Sequence, TypeVar, cast

from pydantic_core import ValidationError

from ..utils import get_variable_name_inspect
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


AgentType = TypeVar("AgentType")  # self-referential type hint


def format_validation_error_msg(e: ValidationError, tool_name: str) -> str:
    error_txt = ""
    for error in e.errors():
        error_txt += (
            f"Tool: {tool_name}\nAttribute: {error['loc'][0]}\n"
            f"Input value: {error['input']}\n{e.__class__.__name__}: {error['msg']}\n\n"
        )
    return error_txt.strip()


class Agent(Generic[ClientMessage, ClientMessageContent, ClientTool]):
    """
    The Agent class is the main class to interact with the model and tools.

    Parameters
    ----------
    client : LLMClient[ClientMessage, ClientMessageContent, ClientTool], optional
        The client instance to interact with the model, by default None.
    llm : LLMModel[ClientMessage, ClientMessageContent, ClientTool], optional
        The LLMModel instance to interact with the model, by default None.
    tools : list[Callable[..., Any] | Tool], optional
        The list of tools to be used in the conversation, by default None.
    subagents : list[Agent], optional
        The list of subagents to be used in the conversation, by default None.
        Subagents are treated as tools in the parent agent, which is prepared dynamically.
        Tool name is generated based on the subagent's variable name, e.g., route_{subagent_variable_name}.
        Please be careful for the conflict of the tool name in the global namespace.
    agent_config : AgentConfig, optional
        The configuration of the agent, by default None.
    conversation_memory : BaseConversationMemory, optional
        The conversation memory to store the conversation history, by default None.
    logger : Logger, optional
        The logger instance to log the information, by default None.
    response_schema_as_tool : BaseModel, optional
        The response schema as a tool to validate the final answer, by default None.
        If provided, then the final answer is structured using tool-calling, and is validated based on the schema.
    **kwargs : Any
        The additional keyword arguments to be passed to the LLMModel instance.
    """

    def __init__(
        self,
        client: LLMClient[ClientMessage, ClientMessageContent, ClientTool] | None = None,
        llm: LLMModel[ClientMessage, ClientMessageContent, ClientTool] | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        subagents: list[AgentType] | None = None,
        agent_config: AgentConfig | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        logger: Logger | None = None,
        response_schema_as_tool: BaseModel | None = None,
        **kwargs: Any,
    ):
        self._client = client
        self.agent_config = agent_config or AgentConfig()
        self.retry_prompt = self.agent_config.internal_prompt.validation_error_retry
        self.conversation_memory = conversation_memory
        self.n_validation_retries = self.agent_config.n_validation_retries
        self.logger = logger or default_logger
        self.response_schema_as_tool = response_schema_as_tool
        self._store_conversation = self.agent_config.store_conversation
        self.__response_schema_name = "final_answer"
        self.__review_prompt = self.agent_config.internal_prompt.review
        self.__max_repeat_text_response = 3
        _tools = tools or []

        for subagent in subagents or []:
            if isinstance(subagent, Agent):
                # If conversation_memory is provided to the subagent, then it's overridden by the,
                # conversation_memory of the main agent.
                subagent.conversation_memory = conversation_memory

                # Internal conversaion history of the subagent isn't stored.
                subagent._store_conversation = False

                _tools += [_generate_dynamic_tool_as_agent(agent=subagent)]
            else:
                raise ValueError(
                    "Subagent must be an instance of Agent class. "
                    "Please provide the correct agent instance."
                )

        if response_schema_as_tool:
            _tools += self._prepare_tool_as_response_schema(response_schema_as_tool)

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

        if _tools is not None:
            self._tools = {tool.name: tool for tool in self.llm._prepare_tools(_tools)}

    def _prepare_tool_as_response_schema(self, response_schema_as_tool: BaseModel) -> list[Tool]:
        _response_schema_json = response_schema_as_tool.model_json_schema()
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

                    if self.response_schema_as_tool is None:
                        if self._include_tool_call_response(response):
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(response):
                            n_repeat_text_response = 0
                        else:
                            n_repeat_text_response += 1
                            messages.append(Prompt(contents=self.__review_prompt, role="user"))

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

                    if self.response_schema_as_tool is None:
                        if self._include_tool_call_response(response):
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(response):
                            n_repeat_text_response = 0
                        else:
                            n_repeat_text_response += 1
                            messages.append(Prompt(contents=self.__review_prompt, role="user"))

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
                # self.store_history(messages)
                final_result.usage = total_usage
                yield final_result
                break
            else:
                if is_error:
                    n_validation_retries += 1
                else:
                    n_validation_retries = 0

                    total_usage = self._update_usage(total_usage, chunk)
                    if self.response_schema_as_tool is None:
                        if self._include_tool_call_response(chunk):
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(chunk):
                            n_repeat_text_response = 0
                        else:
                            n_repeat_text_response += 1
                            messages.append(Prompt(contents=self.__review_prompt, role="user"))

        self.store_history(messages)

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
                # self.store_history(messages)
                final_result.usage = total_usage
                yield final_result
                break
            else:
                if is_error:
                    n_validation_retries += 1
                else:
                    n_validation_retries = 0

                    total_usage = self._update_usage(total_usage, chunk)
                    if self.response_schema_as_tool is None:
                        if self._include_tool_call_response(chunk):
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(chunk):
                            n_repeat_text_response = 0
                        else:
                            n_repeat_text_response += 1
                            messages.append(Prompt(contents=self.__review_prompt, role="user"))

        self.store_history(messages)

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
                            self.response_schema_as_tool is not None
                        ), "Please provide response_schema_as_tool in the agent."
                        self.response_schema_as_tool.model_validate(args)
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
        if self.conversation_memory is not None and self._store_conversation:
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

    def _prepare_response_schema_tool(self, response_schema_as_tool: dict[str, Any]) -> list[Tool]:
        return [
            Tool(
                name=self.__response_schema_name,
                description=(
                    "The final answer which ends this conversation."
                    "Must run at the end of the conversation and "
                    "arguments of the tool must bases on the conversation history, not be made up."
                ),
                schema_dict=response_schema_as_tool,
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


def _get_agent_tools_description(agent: AgentType) -> str:
    return "\n".join(
        [f"- {funcname}: {tool.description}" for funcname, tool in agent._tools.items()]
    )


def _run_subagent(agent: AgentType, instruction: str) -> str:
    """
    This function is used to run the subagent from the main agent.

    Parameters
    ----------
    agent : AgentType
        The subagent instance.
    instruction : str
        The detail and specific instruction to run the subagent, based on the entire conversation history or tool's result.

    Returns
    ----------
    str
        The response from the subagent.
    """
    if not isinstance(agent, Agent):
        raise ValueError(
            "Subagent must be an instance of Agent class. "
            "Please provide the correct agent instance."
        )

    return agent.generate_text(instruction).contents[0].text  # type: ignore


def _duplicate_function(func: Callable[..., Any], name: str) -> Callable[..., Any]:
    new_function_name = name
    new_function_code = func.__code__
    new_function_globals = func.__globals__
    new_function_defaults = func.__defaults__
    new_function_closure = func.__closure__

    new_func = types.FunctionType(
        new_function_code,
        new_function_globals,
        new_function_name,
        new_function_defaults,
        new_function_closure,
    )
    new_func.__annotations__ = func.__annotations__
    globals()[new_function_name] = new_func
    return new_func


def _generate_dynamic_tool_as_agent(agent: AgentType) -> Tool:
    agent_name = get_variable_name_inspect(agent)
    duplicated_tool = _duplicate_function(_run_subagent, f"route_{agent_name}")
    agent_description = _get_agent_tools_description(agent)

    return Tool(
        tool=duplicated_tool,
        name=f"route_{agent_name}",
        description=(
            f"This function invokes the agent capable to run the following tools:\n"
            f"{agent_description}"
        ),
        context={"agent": agent},
    )
