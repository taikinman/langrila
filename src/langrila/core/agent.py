import json
import types
from logging import Logger
from typing import Any, AsyncGenerator, Callable, Generator, Generic, Sequence, TypeVar

from pydantic_core import ValidationError

from ..utils import get_variable_name_inspect
from .client import LLMClient
from .config import AgentConfig
from .embedding import EmbeddingResults
from .logger import DEFAULT_LOGGER as default_logger
from .memory import BaseConversationMemory
from .message import Message
from .model import LLMModel
from .prompt import Prompt, PromptType, SystemPrompt, TextPrompt, ToolUsePrompt
from .pydantic import BaseModel
from .response import Response, ResponseType, TextResponse, ToolCallResponse
from .tool import Tool
from .typing import ClientMessage, ClientMessageContent, ClientSystemMessage, ClientTool
from .usage import NamedUsage, Usage

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


class Agent(Generic[ClientMessage, ClientSystemMessage, ClientMessageContent, ClientTool]):
    """
    The Agent class is the main class to interact with the model and tools.

    Parameters
    ----------
    client : LLMClient[ClientMessage, ClientSystemMessage, ClientMessageContent, ClientTool], optional
        The client instance to interact with the model, by default None.
    llm : LLMModel[ClientMessage, ClientSystemMessage, ClientMessageContent, ClientTool], optional
        The LLMModel instance to interact with the model, by default None.
    tools : list[Callable[..., Any] | Tool], optional
        The list of tools to be used in the conversation, by default None.
        Schema of the tool is generated based on the function signature and docstring.
        If you want to inject context to the tool, please specify Tool instance instead of
        function itself.
    subagents : list[Agent], optional
        The list of subagents to be used in the conversation, by default None.
        Subagents are treated as tools in the parent agent, which is prepared dynamically.
        Tool name is generated based on the subagent's variable name, e.g., route_{subagent_variable_name}.
        Please be careful for the conflict of the tool name in the global namespace.
    agent_config : AgentConfig, optional
        The internal configuration of the agent, by default None.
    conversation_memory : BaseConversationMemory, optional
        The conversation memory to store the conversation history, by default None.
    logger : Logger, optional
        The logger instance to log the information, by default None.
    response_schema_as_tool : BaseModel, optional
        The response schema as a tool to validate the final answer, by default None.
        If provided, then the final answer is structured using tool-calling, and is validated based on the schema.
    system_instruction : SystemPrompt, optional
        The system instruction. If provided, then it's used as a system instruction in the conversation, by default None.
    **kwargs : Any
        The additional keyword arguments to be passed to the LLMModel instance.
    """

    def __init__(
        self,
        client: LLMClient[ClientMessage, ClientSystemMessage, ClientMessageContent, ClientTool]
        | None = None,
        llm: LLMModel[ClientMessage, ClientSystemMessage, ClientMessageContent, ClientTool]
        | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        subagents: list[AgentType] | None = None,
        agent_config: AgentConfig | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        logger: Logger | None = None,
        response_schema_as_tool: BaseModel | None = None,
        system_instruction: SystemPrompt | None = None,
        **kwargs: Any,
    ):
        self._client = client
        self._subagents = subagents
        self.agent_config = agent_config or AgentConfig()
        self.retry_prompt = self.agent_config.internal_prompt.error_retry
        self.conversation_memory = conversation_memory
        self.n_validation_retries = self.agent_config.n_validation_retries
        self.logger = logger or default_logger
        self.response_schema_as_tool = response_schema_as_tool
        self.system_instruction = (
            self._setup_system_instruction(system_instruction, self.agent_config)
            if system_instruction
            else self.agent_config.internal_prompt.system_instruction
        )

        self._store_conversation = self.agent_config.store_conversation
        self.__response_schema_name = "final_answer"
        self.__no_tool_use_retry_prompt = self.agent_config.internal_prompt.no_tool_use_retry
        self.__max_repeat_text_response = 3
        _tools = tools or []
        self._name = "root"

        for subagent in subagents or []:
            if isinstance(subagent, Agent):
                self._recurse_setup_subagent(subagent)
                agent_name = get_variable_name_inspect(subagent)
                subagent._name = agent_name
                _tools += [
                    self._generate_dynamic_tool_as_agent(agent=subagent, agent_name=agent_name)
                ]
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

            self.llm.tools += _tools
        else:
            assert client is not None, "Either client or llm must be provided"

            self.llm = LLMModel(
                client=client,
                logger=logger,
                system_instruction=system_instruction,
                tools=_tools,
                **kwargs,
            )

        self.tools = _tools
        self.__max_tool_call_retry = len(_tools) + 2

    def _gather_subagent_usage(self) -> NamedUsage:
        all_usages = NamedUsage()
        for subagent in self._subagents or []:
            if isinstance(subagent, Agent):
                all_usages += subagent._gather_subagent_usage()

                if hasattr(subagent, "_usage"):
                    all_usages += subagent._usage
                    subagent._usage = NamedUsage()  # reset to avoid duplication
            else:
                raise ValueError(
                    "Subagent must be an instance of Agent class. "
                    "Please provide the correct agent instance."
                )

        return all_usages

    def _recurse_setup_subagent(self, subagent: "Agent") -> None:
        def _setup_subagent(subagent: "Agent") -> None:
            # If conversation_memory is provided to the subagent, then it's overridden by the,
            # conversation_memory of the orchestrator agent.
            subagent.conversation_memory = self.conversation_memory

            # Internal conversaion history of the subagent isn't stored but input.
            subagent._store_conversation = False

            # override logger of the subagent
            subagent.logger = subagent.llm.logger = self.logger

        _setup_subagent(subagent)

        for _subagent in subagent._subagents or []:
            if isinstance(_subagent, Agent):
                self._recurse_setup_subagent(_subagent)
            else:
                raise ValueError(
                    "Subagent must be an instance of Agent class. "
                    "Please provide the correct agent instance."
                )

    def _setup_system_instruction(
        self,
        system_instruction: SystemPrompt,
        agent_config: AgentConfig,
    ) -> SystemPrompt:
        _system_instruction = (
            agent_config.internal_prompt.system_instruction + "\n\n" + system_instruction.contents
        )
        return SystemPrompt(
            role=system_instruction.role,
            contents=_system_instruction.strip(),
            name=system_instruction.name,
        )

    def _prepare_tool_as_response_schema(self, response_schema_as_tool: BaseModel) -> list[Tool]:
        _response_schema_json = response_schema_as_tool.model_json_schema()
        return self._prepare_response_schema_tool(_response_schema_json)

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

    def _get_tools_dict(
        self,
        tools: list[Callable[..., Any] | Tool],
    ) -> dict[str, Tool]:
        outputs = {}
        for tool in self.llm._prepare_tools(tools):
            if tool.name:
                outputs[tool.name] = tool
            else:
                raise ValueError("Tool name is required.")
        return outputs

    def _update_usage(self, base_usage: NamedUsage, response: Response) -> NamedUsage:
        if isinstance(response.usage, NamedUsage):
            base_usage += response.usage
        elif isinstance(response.usage, Usage):
            if response.name:
                if response.name in base_usage:
                    base_usage[response.name] += response.usage
                else:
                    base_usage[response.name] = response.usage
            else:
                if "root" in base_usage:
                    base_usage["root"] += response.usage
                else:
                    base_usage["root"] = response.usage
        else:
            raise ValueError("Usage must be either NamedUsage or Usage.")

        return base_usage

    def generate_text(
        self,
        prompt: AgentInput,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        **kwargs: Any,
    ) -> Response:
        self._usage = NamedUsage()
        _tools = tools or self.tools
        _tools_dict = self._get_tools_dict(_tools)

        messages = self.load_history()
        messages.extend(self._process_user_prompt(prompt))

        n_validation_retries = 0
        n_repeat_text_response = 0
        n_repeat_tool_call = 0
        final_result = None
        while (
            n_validation_retries < self.n_validation_retries
            and n_repeat_text_response < self.__max_repeat_text_response
            and n_repeat_tool_call < self.__max_tool_call_retry
        ):
            response = self.llm.generate_text(
                messages=messages,
                system_instruction=system_instruction,
                tools=_tools,
                **kwargs,
            )

            self._usage = self._update_usage(self._usage, response)
            self._usage += self._gather_subagent_usage()

            if not response.contents:
                self.logger.error("No response received from the model. Retrying.")
                continue

            messages.append(response)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(response, _tools_dict)
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                self.logger.debug(f"Final result: {final_result.contents}")
                messages.append(final_result)
                break
            else:
                if is_error:
                    n_validation_retries += 1
                    n_repeat_tool_call += 1
                    messages.append(
                        Prompt(contents=[TextPrompt(text=self.retry_prompt)], role="user")
                    )
                else:
                    n_validation_retries = 0

                    if self.response_schema_as_tool is None:
                        if self._include_tool_call_response(response):
                            n_repeat_text_response = 0
                            n_repeat_tool_call += 1
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(response):
                            n_repeat_text_response = 0
                            n_repeat_tool_call += 1
                        else:
                            n_repeat_text_response += 1
                            n_repeat_tool_call = 0

                            if n_repeat_text_response != self.__max_repeat_text_response:
                                messages.append(
                                    Prompt(contents=self.__no_tool_use_retry_prompt, role="user")
                                )

        self.store_history(messages)

        if final_result:
            final_result.usage = self._usage
            return final_result
        else:
            response.usage = self._usage
            return response

    async def generate_text_async(
        self,
        prompt: AgentInput,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        **kwargs: Any,
    ) -> Response:
        self._usage = NamedUsage()
        _tools = tools or self.tools
        _tools_dict = self._get_tools_dict(_tools)

        messages = self.load_history()
        messages.extend(self._process_user_prompt(prompt))

        n_validation_retries = 0
        n_repeat_text_response = 0
        n_repeat_tool_call = 0
        final_result = None
        while (
            n_validation_retries < self.n_validation_retries
            and n_repeat_text_response < self.__max_repeat_text_response
            and n_repeat_tool_call < self.__max_tool_call_retry
        ):
            response = await self.llm.generate_text_async(
                messages=messages,
                system_instruction=system_instruction,
                tools=_tools,
                **kwargs,
            )

            self._usage = self._update_usage(self._usage, response)
            self._usage += self._gather_subagent_usage()

            if not response.contents:
                self.logger.error("No response received from the model. Retrying.")
                continue

            messages.append(response)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(response, _tools_dict)
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                self.logger.debug(f"Final result: {final_result.contents}")
                messages.append(final_result)
                break
            else:
                if is_error:
                    n_validation_retries += 1
                    n_repeat_tool_call += 1
                    messages.append(
                        Prompt(contents=[TextPrompt(text=self.retry_prompt)], role="user")
                    )
                else:
                    n_validation_retries = 0

                    if self.response_schema_as_tool is None:
                        if self._include_tool_call_response(response):
                            n_repeat_text_response = 0
                            n_repeat_tool_call += 1
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(response):
                            n_repeat_text_response = 0
                            n_repeat_tool_call += 1
                        else:
                            n_repeat_text_response += 1
                            n_repeat_tool_call = 0

                            if n_repeat_text_response != self.__max_repeat_text_response:
                                messages.append(
                                    Prompt(contents=self.__no_tool_use_retry_prompt, role="user")
                                )

        self.store_history(messages)

        if final_result:
            final_result.usage = self._usage
            return final_result
        else:
            response.usage = self._usage
            return response

    def stream_text(
        self,
        prompt: AgentInput,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        **kwargs: Any,
    ) -> Generator[Response, None, None]:
        self._usage = NamedUsage()
        _tools = tools or self.tools
        _tools_dict = self._get_tools_dict(_tools)

        messages = self.load_history()
        messages.extend(self._process_user_prompt(prompt))

        n_validation_retries = 0
        n_repeat_text_response = 0
        n_repeat_tool_call = 0
        final_result = None
        while (
            n_validation_retries < self.n_validation_retries
            and n_repeat_text_response < self.__max_repeat_text_response
            and n_repeat_tool_call < self.__max_tool_call_retry
        ):
            streamed_response = self.llm.stream_text(
                messages=messages,
                system_instruction=system_instruction,
                tools=_tools,
                **kwargs,
            )
            for chunk in streamed_response:
                if not chunk.is_last_chunk:
                    yield chunk

            messages.append(chunk)

            self._usage = self._update_usage(self._usage, chunk)
            self._usage += self._gather_subagent_usage()

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(chunk, _tools_dict)
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                self.logger.debug(f"Final result: {final_result.contents}")
                messages.append(final_result)
                # self.store_history(messages)
                final_result.usage = self._usage
                yield final_result
                break
            else:
                if is_error:
                    n_validation_retries += 1
                    n_repeat_tool_call += 1
                    messages.append(
                        Prompt(contents=[TextPrompt(text=self.retry_prompt)], role="user")
                    )
                else:
                    n_validation_retries = 0

                    if self.response_schema_as_tool is None:
                        if self._include_tool_call_response(chunk):
                            n_repeat_text_response = 0
                            n_repeat_tool_call += 1
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(chunk):
                            n_repeat_text_response = 0
                            n_repeat_tool_call += 1
                        else:
                            n_repeat_text_response += 1
                            n_repeat_tool_call = 0

                            if n_repeat_text_response != self.__max_repeat_text_response:
                                messages.append(
                                    Prompt(contents=self.__no_tool_use_retry_prompt, role="user")
                                )

        self.store_history(messages)

        chunk.usage = self._usage
        yield chunk

    async def stream_text_async(
        self,
        prompt: AgentInput,
        system_instruction: SystemPrompt | None = None,
        tools: list[Callable[..., Any] | Tool] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Response, None]:
        self._usage = NamedUsage()
        _tools = tools or self.tools
        _tools_dict = self._get_tools_dict(_tools)

        messages = self.load_history()
        messages.extend(self._process_user_prompt(prompt))

        n_validation_retries = 0
        n_repeat_text_response = 0
        n_repeat_tool_call = 0
        final_result = None
        while (
            n_validation_retries < self.n_validation_retries
            and n_repeat_text_response < self.__max_repeat_text_response
            and n_repeat_tool_call < self.__max_tool_call_retry
        ):
            streamed_response = self.llm.stream_text_async(
                messages=messages,
                system_instruction=system_instruction,
                tools=_tools,
                **kwargs,
            )
            async for chunk in streamed_response:
                if not chunk.is_last_chunk:
                    yield chunk

            messages.append(chunk)

            self._usage = self._update_usage(self._usage, chunk)
            self._usage += self._gather_subagent_usage()

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(chunk, _tools_dict)
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                self.logger.debug(f"Final result: {final_result.contents}")
                messages.append(final_result)
                # self.store_history(messages)
                final_result.usage = self._usage
                yield final_result
                break
            else:
                if is_error:
                    n_validation_retries += 1
                    n_repeat_tool_call += 1
                    messages.append(
                        Prompt(contents=[TextPrompt(text=self.retry_prompt)], role="user")
                    )
                else:
                    n_validation_retries = 0

                    if self.response_schema_as_tool is None:
                        if self._include_tool_call_response(chunk):
                            n_repeat_text_response = 0
                            n_repeat_tool_call += 1
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(chunk):
                            n_repeat_text_response = 0
                            n_repeat_tool_call += 1
                        else:
                            n_repeat_text_response += 1
                            n_repeat_tool_call = 0

                            if n_repeat_text_response != self.__max_repeat_text_response:
                                messages.append(
                                    Prompt(contents=self.__no_tool_use_retry_prompt, role="user")
                                )

        self.store_history(messages)

        chunk.usage = self._usage
        yield chunk

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
        self, response: Response, tools_dict: dict[str, Tool]
    ) -> tuple[Prompt | None, bool, Response | None]:
        next_turn_contents: list[PromptType] = []
        is_error = False
        final_result = None
        for content in response.contents or []:
            if isinstance(content, ToolCallResponse):
                tool_name = content.name
                if not tool_name:
                    raise ValueError("Tool name is required.")

                try:
                    tool = tools_dict[tool_name]
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

                        # validate response schema
                        self.response_schema_as_tool.model_validate(args)

                        next_turn_contents.append(
                            ToolUsePrompt(
                                output=json.dumps(args, ensure_ascii=False),
                                call_id=content.call_id,
                                args=content.args,
                                name=tool_name,
                            )
                        )

                        final_result = Response(
                            role="assistant",
                            contents=[TextResponse(text=json.dumps(args, ensure_ascii=False))],
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
                    "arguments of the tool must be picked from the conversation history, not be made up."
                ),
                schema_dict=response_schema_as_tool,
            )
        ]

    def generate_image(self, prompt: str, **kwargs: Any) -> Response:
        return self.llm.generate_image(prompt, **kwargs)

    async def generate_image_async(self, prompt: str, **kwargs: Any) -> Response:
        return await self.llm.generate_image_async(prompt, **kwargs)

    def generate_audio(self, prompt: AgentInput, **kwargs: Any) -> Response:
        messages = self.load_history()
        messages.extend(self._process_user_prompt(prompt))
        response = self.llm.generate_audio(messages, **kwargs)

        messages.append(response)
        self.store_history(messages)
        return response

    async def generate_audio_async(self, prompt: AgentInput, **kwargs: Any) -> Response:
        messages = self.load_history()

        messages.extend(self._process_user_prompt(prompt))
        response = await self.llm.generate_audio_async(messages, **kwargs)

        messages.append(response)
        self.store_history(messages)
        return response

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        return self.llm.embed_text(texts, **kwargs)

    async def embed_text_async(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        return await self.llm.embed_text_async(texts, **kwargs)

    def _generate_dynamic_tool_as_agent(self, agent: AgentType, agent_name: str) -> Tool:
        tool_name = f"route_{agent_name}"
        if tool_name in globals():
            self.logger.warning(
                (
                    f"Function: {tool_name} already exists in the global namespace. "
                    "Skipping routing tool creation."
                )
            )
            duplicated_tool = globals()[tool_name]
        else:
            duplicated_tool = _duplicate_function(_run_subagent, tool_name)

        agent_description = _get_agent_tools_description(agent)

        return Tool(
            tool=duplicated_tool,
            name=tool_name,
            description=(
                f"This function invokes the agent capable to run the following tools:\n"
                f"{agent_description}"
            ),
            context={"agent": agent, "agent_name": agent_name},
        )


def _get_agent_tools_description(agent: AgentType) -> str:
    if not isinstance(agent, Agent):
        raise ValueError(
            "Subagent must be an instance of Agent class. "
            "Please provide the correct agent instance."
        )

    return "\n".join(
        [f"- {tool.name}: {tool.description}" for tool in agent.llm._prepare_tools(agent.tools)]
    )


def _run_subagent(agent: AgentType, agent_name: str, instruction: str) -> str:
    """
    This function is used to run the subagent from the parent agent.

    Parameters
    ----------
    agent : AgentType
        The subagent instance.
    agent_name : str
        The name of the subagent.
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

    return agent.generate_text(instruction, name=agent_name).contents[0].text  # type: ignore


def _duplicate_function(func: Callable[..., Any], name: str) -> Callable[..., Any]:
    """
    This function dinamically duplicates the function with a new name.
    """
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
