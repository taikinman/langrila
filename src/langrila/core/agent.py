import copy
import inspect
import json
import types
from dataclasses import dataclass
from logging import Logger
from typing import Any, AsyncGenerator, Callable, Generator, Generic, Sequence, TypeVar, final

from pydantic_core import ValidationError

from ..utils import get_variable_name_inspect
from ._context import AgentInternalContext
from .client import LLMClient
from .config import AgentConfig
from .embedding import EmbeddingResults
from .error import RetryLimitExceededError
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
    str
    | Prompt
    | PromptType
    | Response
    | ResponseType
    | list[str | Prompt | PromptType | ResponseType | Response]
)


AgentType = TypeVar("AgentType")  # self-referential type hint


def format_validation_error_msg(e: ValidationError, tool_name: str) -> str:
    error_txt = ""
    for error in e.errors():
        error_txt += (
            f"Tool: {tool_name}\nAttribute: {error['loc'][0]}\n"
            f"Input value: {error['input']}\n"
            f"{e.__class__.__name__}: {error['msg']}\n\n"
        )
    return error_txt.strip()


@final
@dataclass(init=False)  # to apply pydantic TypeAdapter
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
    planning : bool, optional
        If True, the agent makes a plan to answer the user's question/requirement at the first step, by default False.
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
        subagents: list["Agent"] | None = None,  # type: ignore
        agent_config: AgentConfig | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        logger: Logger | None = None,
        response_schema_as_tool: BaseModel | None = None,
        system_instruction: SystemPrompt | None = None,
        planning: bool = False,
        **kwargs: Any,
    ):
        self._client = client
        self.response_schema_as_tool = response_schema_as_tool
        self.subagents = subagents or []
        self.conversation_memory = conversation_memory
        self.planning = planning
        self.agent_config = agent_config or AgentConfig()
        self.retry_prompt = self.agent_config.internal_prompt.error_retry
        self.max_error_retries = self.agent_config.max_error_retries
        self.logger = logger or default_logger
        self.system_instruction = system_instruction
        self._store_conversation = self.agent_config.store_conversation
        self.__response_schema_name = "final_answer"
        self.__no_tool_use_retry_prompt = self.agent_config.internal_prompt.no_tool_use_retry
        self._name = "root"
        self._usage: NamedUsage

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
                system_instruction=system_instruction,
                # **kwargs,
            )

        self.tools = self._setup_all_tools(
            tools=tools,
            subagents=subagents,
        )

        self.init_kwargs = kwargs

    def _gather_subagent_usage(self) -> NamedUsage:
        """Sub-agent is invoked as a tool in the parent agent.
        Tool's output is just string object, so it doesn't have usage.
        This function is used to gather the usage of the subagent.
        """
        all_usages = NamedUsage()
        for subagent in self.subagents or []:
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

    def _recurse_setup_subagent(
        self,
        subagent: "Agent",  # type: ignore
    ) -> None:
        def _setup_subagent(subagent: "Agent") -> None:  # type: ignore
            # If conversation_memory is provided to the subagent, then it's overridden by the,
            # conversation_memory of the orchestrator agent.
            subagent.conversation_memory = self.conversation_memory

            # Internal conversaion history of the subagent isn't stored but input.
            subagent._store_conversation = False

            # override logger of the subagent
            subagent.logger = subagent.llm.logger = self.logger

        _setup_subagent(subagent)

        for _subagent in subagent.subagents or []:
            if isinstance(_subagent, Agent):
                self._recurse_setup_subagent(_subagent)
            else:
                raise ValueError(
                    "Subagent must be an instance of Agent class. "
                    "Please provide the correct agent instance."
                )

    def _prepare_tool_as_response_schema(self, response_schema_as_tool: BaseModel) -> list[Tool]:
        _response_schema_json = response_schema_as_tool.model_json_schema()
        return self._prepare_response_schema_tool(_response_schema_json)

    def _include_tool_call_response(self, response: Response) -> bool:
        return any([isinstance(content, ToolCallResponse) for content in response.contents])

    def _get_tools_dict(
        self,
        tools: list[Tool],
    ) -> dict[str, Tool]:
        outputs = {}
        for tool in tools:
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

        usage_subagent = self._gather_subagent_usage()
        return base_usage + usage_subagent

    def _make_planning_prompt_contnet(self, prompt: AgentInput) -> list[TextPrompt]:
        processed_prompt = self._process_user_prompt(prompt)

        user_input = "\n".join(
            [
                content.text
                for _prompt in processed_prompt
                for content in _prompt.contents
                if isinstance(content, TextPrompt)
            ]
        )

        if not user_input:
            raise ValueError("Must includes text prompt in the user input for planning.")

        return [
            TextPrompt(
                text=self.agent_config.internal_prompt.planning.format(
                    user_input=user_input,
                    capabilities=_get_agent_capabilities(self),
                )
            )
        ]

    def _validate_generation_params(self, **kwargs: Any) -> None:
        params_constructor_only = {
            "tools",
            "subagents",
            "response_schema_as_tool",
            "logger",
            "agent_config",
        }

        for key in kwargs:
            if key in params_constructor_only:
                raise ValueError(f"Invalid parameter: {key} must be provide in the constructor.")

    def _setup_all_tools(
        self,
        tools: list[Callable[..., Any] | Tool] | None = None,
        subagents: list["Agent"] | None = None,  # type: ignore
    ) -> list[Tool]:
        _tools = copy.copy(tools) or []
        _subagents = subagents or []

        for subagent in _subagents:
            if isinstance(subagent, Agent):
                self._recurse_setup_subagent(subagent)
                agent_name = get_variable_name_inspect(subagent)
                subagent._name = agent_name
                _tools += [_generate_dynamic_tool_as_agent(agent=subagent, agent_name=agent_name)]
            else:
                raise ValueError(
                    "Subagent must be an instance of Agent class. "
                    "Please provide the correct agent instance."
                )

        _tools = self.llm._prepare_tools(_tools)

        return _tools

    def _filter_kwargs_for_planning(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        remove_keys = {"response_schema", "response_format"}
        return {k: v for k, v in kwargs.items() if k not in remove_keys}

    def _planning_step(
        self,
        messages: list[Prompt | Response],
        prompt: AgentInput,
        system_instruction: SystemPrompt | None = None,
        **kwargs: Any,
    ) -> list[Prompt | Response]:
        while True:
            # Provide information about subagents and tools to the agnet
            planning_message = Prompt(
                contents=self._make_planning_prompt_contnet(prompt), role="user"
            )

            # planning without tools
            response = self.llm.generate_text(
                messages=messages + [planning_message],
                system_instruction=system_instruction,
                **self._filter_kwargs_for_planning(**kwargs),
            )

            self._usage = self._update_usage(self._usage, response)

            if not response.contents:
                self.logger.error("No response received from the model. Retrying.")
                continue

            # Next turn message is configurable internal prompt
            next_turn_message = Prompt(
                contents=[TextPrompt(text=self.agent_config.internal_prompt.do_plan)],
                role="user",
            )

            break

        return [planning_message, response, next_turn_message]

    async def _planning_step_async(
        self,
        messages: list[Prompt | Response],
        prompt: AgentInput,
        system_instruction: SystemPrompt | None = None,
        **kwargs: Any,
    ) -> list[Prompt | Response]:
        while True:
            # Provide information about subagents and tools to the agnet
            planning_message = Prompt(
                contents=self._make_planning_prompt_contnet(prompt), role="user"
            )

            # planning without tools
            response = await self.llm.generate_text_async(
                messages=messages + [planning_message],
                system_instruction=system_instruction,
                **self._filter_kwargs_for_planning(**kwargs),
            )

            self._usage = self._update_usage(self._usage, response)

            if not response.contents:
                self.logger.error("No response received from the model. Retrying.")
                continue

            # Next turn message is configurable internal prompt
            next_turn_message = Prompt(
                contents=[TextPrompt(text=self.agent_config.internal_prompt.do_plan)],
                role="user",
            )

            break

        return [planning_message, response, next_turn_message]

    def generate_text(
        self,
        prompt: AgentInput,
        system_instruction: SystemPrompt | None = None,
        response_schema_as_tool: BaseModel | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate text response from the model. If you specified the same parameter in both
        the constructor and this method, the parameter in this method will be used.

        Parameters
        ----------
        prompt : AgentInput
            The user input to the agent.
        system_instruction : SystemPrompt, optional
            The system instruction to the agent, by default None.
        response_schema_as_tool : BaseModel, optional
            The response schema as a tool to validate the final answer, by default None.
            If provided, then the final answer is structured using tool-calling, and is validated based on the schema.
        **kwargs : Any
            Additional keyword arguments to pass to the API provider.
            Basically the same as the parameters in the original provider API.
            It means the agent accepts original parameters of OpenAI API if you use OpenAIClient in this agent.
            For more details, please refer to the API reference of each provider.

        Returns
        ----------
        Response
            The response from the model containing TextResponse. You can access the response text
            as `response.contents[0].text`.
        """
        all_kwargs = {**self.init_kwargs, **kwargs}
        self._validate_generation_params(**all_kwargs)

        self._usage = NamedUsage()

        _tools = copy.copy(self.tools)
        _response_schema_as_tool = response_schema_as_tool or self.response_schema_as_tool
        if _response_schema_as_tool:
            _tools += self._prepare_tool_as_response_schema(_response_schema_as_tool)
        _tools_dict = self._get_tools_dict(_tools)

        messages = self.load_history()

        if self.planning:
            messages.extend(
                self._planning_step(
                    messages=messages,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    **all_kwargs,
                )
            )
        else:
            messages.extend(self._process_user_prompt(prompt))

        ctx = AgentInternalContext(
            max_error_retries=self.max_error_retries,
            max_repeat_tool_call=len(_tools) + 2,
        )
        final_result = None
        while ctx:
            response = self.llm.generate_text(
                messages=messages,
                system_instruction=system_instruction,
                tools=_tools,
                **all_kwargs,
            )

            self._usage = self._update_usage(self._usage, response)

            if not response.contents:
                self.logger.error("No response received from the model. Retrying.")
                continue

            messages.append(response)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(
                    response, _tools_dict, _response_schema_as_tool
                )
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                self.logger.debug(f"Final result: {final_result.contents}")
                messages.append(final_result)
                break
            else:
                if is_error:
                    ctx.increment_error_retries_count()
                    ctx.increment_repeat_tool_call_count()
                    messages.append(
                        Prompt(contents=[TextPrompt(text=self.retry_prompt)], role="user")
                    )
                else:
                    ctx.reset_error_retries_count()

                    if _response_schema_as_tool is None:
                        if self._include_tool_call_response(response):
                            ctx.increment_repeat_tool_call_count()
                            ctx.reset_repeat_text_response_count()
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(response):
                            ctx.reset_repeat_text_response_count()
                            ctx.increment_repeat_tool_call_count()
                        else:
                            ctx.increment_repeat_text_response_count()
                            ctx.reset_repeat_tool_call_count()

                            if ctx.text_response_count != ctx.max_repeat_text_response:
                                messages.append(
                                    Prompt(contents=self.__no_tool_use_retry_prompt, role="user")
                                )

        if not ctx:
            raise RetryLimitExceededError()

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
        response_schema_as_tool: BaseModel | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate text response from the model asynchronously. If you specified the same parameter in both
        the constructor and this method, the parameter in this method will be used.


        Parameters
        ----------
        prompt : AgentInput
            The user input to the agent.
        system_instruction : SystemPrompt, optional
            The system instruction to the agent, by default None.
        response_schema_as_tool : BaseModel, optional
            The response schema as a tool to validate the final answer, by default None.
            If provided, then the final answer is structured using tool-calling, and is validated based on the schema.
        **kwargs : Any
            Additional keyword arguments to pass to the API provider.
            Basically the same as the parameters in the original provider API.
            It means the agent accepts original parameters of OpenAI API if you use OpenAIClient in this agent.
            For more details, please refer to the API reference of each provider.

        Returns
        ----------
        Response
            The response from the model containing TextResponse. You can access the response text
            as `response.contents[0].text`.
        """
        all_kwargs = {**self.init_kwargs, **kwargs}
        self._validate_generation_params(**all_kwargs)

        self._usage = NamedUsage()

        _tools = copy.copy(self.tools)
        _response_schema_as_tool = response_schema_as_tool or self.response_schema_as_tool
        if _response_schema_as_tool:
            _tools += self._prepare_tool_as_response_schema(_response_schema_as_tool)
        _tools_dict = self._get_tools_dict(_tools)

        messages = self.load_history()

        if self.planning:
            messages.extend(
                await self._planning_step_async(
                    messages=messages,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    **all_kwargs,
                )
            )
        else:
            messages.extend(self._process_user_prompt(prompt))

        ctx = AgentInternalContext(
            max_error_retries=self.max_error_retries,
            max_repeat_tool_call=len(_tools) + 2,
        )
        final_result = None
        while ctx:
            response = await self.llm.generate_text_async(
                messages=messages,
                system_instruction=system_instruction,
                tools=_tools,
                **all_kwargs,
            )

            self._usage = self._update_usage(self._usage, response)

            if not response.contents:
                self.logger.error("No response received from the model. Retrying.")
                continue

            messages.append(response)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(
                    response, _tools_dict, _response_schema_as_tool
                )
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                self.logger.debug(f"Final result: {final_result.contents}")
                messages.append(final_result)
                break
            else:
                if is_error:
                    ctx.increment_error_retries_count()
                    ctx.increment_repeat_tool_call_count()
                    messages.append(
                        Prompt(contents=[TextPrompt(text=self.retry_prompt)], role="user")
                    )
                else:
                    ctx.reset_error_retries_count()

                    if _response_schema_as_tool is None:
                        if self._include_tool_call_response(response):
                            ctx.increment_repeat_tool_call_count()
                            ctx.reset_repeat_text_response_count()
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(response):
                            ctx.reset_repeat_text_response_count()
                            ctx.increment_repeat_tool_call_count()
                        else:
                            ctx.increment_repeat_text_response_count()
                            ctx.reset_repeat_tool_call_count()

                            if ctx.text_response_count != ctx.max_repeat_text_response:
                                messages.append(
                                    Prompt(contents=self.__no_tool_use_retry_prompt, role="user")
                                )

        if not ctx:
            raise RetryLimitExceededError()

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
        response_schema_as_tool: BaseModel | None = None,
        **kwargs: Any,
    ) -> Generator[Response, None, None]:
        """Stream text response from the model. If you specified the same parameter in both
        the constructor and this method, the parameter in this method will be used.


        Parameters
        ----------
        prompt : AgentInput
            The user input to the agent.
        system_instruction : SystemPrompt, optional
            The system instruction to the agent, by default None.
        response_schema_as_tool : BaseModel, optional
            The response schema as a tool to validate the final answer, by default None.
            If provided, then the final answer is structured using tool-calling, and is validated based on the schema.
        **kwargs : Any
            Additional keyword arguments to pass to the API provider.
            Basically the same as the parameters in the original provider API.
            It means the agent accepts original parameters of OpenAI API if you use OpenAIClient in this agent.
            For more details, please refer to the API reference of each provider.

        Yields
        ----------
        Response
            The response from the model. Each chunk is the union of chunks up to that point.
            This method streams TextResponse and ToolCallResponse.
            The last chunk is the same as the result of generate_text method, and
            it's stored in the conversation memory.
        """
        all_kwargs = {**self.init_kwargs, **kwargs}
        self._validate_generation_params(**all_kwargs)

        self._usage = NamedUsage()

        _tools = copy.copy(self.tools)
        _response_schema_as_tool = response_schema_as_tool or self.response_schema_as_tool
        if _response_schema_as_tool:
            _tools += self._prepare_tool_as_response_schema(_response_schema_as_tool)
        _tools_dict = self._get_tools_dict(_tools)

        messages = self.load_history()

        if self.planning:
            messages.extend(
                self._planning_step(
                    messages=messages,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    **all_kwargs,
                )
            )
        else:
            messages.extend(self._process_user_prompt(prompt))

        ctx = AgentInternalContext(
            max_error_retries=self.max_error_retries,
            max_repeat_tool_call=len(_tools) + 2,
        )
        final_result = None
        while ctx:
            streamed_response = self.llm.stream_text(
                messages=messages,
                system_instruction=system_instruction,
                tools=_tools,
                **all_kwargs,
            )
            for chunk in streamed_response:
                if not chunk.is_last_chunk:
                    yield chunk

            messages.append(chunk)

            self._usage = self._update_usage(self._usage, chunk)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(
                    chunk, _tools_dict, _response_schema_as_tool
                )
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                self.logger.debug(f"Final result: {final_result.contents}")
                messages.append(final_result)
                final_result.usage = self._usage
                yield final_result
                break
            else:
                if is_error:
                    ctx.increment_error_retries_count()
                    ctx.increment_repeat_tool_call_count()
                    messages.append(
                        Prompt(contents=[TextPrompt(text=self.retry_prompt)], role="user")
                    )
                else:
                    ctx.reset_error_retries_count()

                    if _response_schema_as_tool is None:
                        if self._include_tool_call_response(chunk):
                            ctx.increment_repeat_tool_call_count()
                            ctx.reset_repeat_text_response_count()
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(chunk):
                            ctx.reset_repeat_text_response_count()
                            ctx.increment_repeat_tool_call_count()
                        else:
                            ctx.increment_repeat_text_response_count()
                            ctx.reset_repeat_tool_call_count()

                            if ctx.text_response_count != ctx.max_repeat_text_response:
                                messages.append(
                                    Prompt(contents=self.__no_tool_use_retry_prompt, role="user")
                                )

        if not ctx:
            raise RetryLimitExceededError()

        self.store_history(messages)

        chunk.usage = self._usage
        yield chunk

    async def stream_text_async(
        self,
        prompt: AgentInput,
        system_instruction: SystemPrompt | None = None,
        response_schema_as_tool: BaseModel | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Response, None]:
        """Stream text response from the model asynchronously. If you specified the same parameter in both
        the constructor and this method, the parameter in this method will be used.


        Parameters
        ----------
        prompt : AgentInput
            The user input to the agent.
        system_instruction : SystemPrompt, optional
            The system instruction to the agent, by default None.
        response_schema_as_tool : BaseModel, optional
            The response schema as a tool to validate the final answer, by default None.
            If provided, then the final answer is structured using tool-calling, and is validated based on the schema.
        **kwargs : Any
            Additional keyword arguments to pass to the API provider.
            Basically the same as the parameters in the original provider API.
            It means the agent accepts original parameters of OpenAI API if you use OpenAIClient in this agent.
            For more details, please refer to the API reference of each provider.

        Yields
        ----------
        Response
            The response from the model. Each chunk is the union of chunks up to that point.
            This method streams TextResponse and ToolCallResponse.
            The last chunk is the same as the result of generate_text method, and
            it's stored in the conversation memory.
        """
        all_kwargs = {**self.init_kwargs, **kwargs}
        self._validate_generation_params(**all_kwargs)

        self._usage = NamedUsage()

        _tools = copy.copy(self.tools)
        _response_schema_as_tool = response_schema_as_tool or self.response_schema_as_tool
        if _response_schema_as_tool:
            _tools += self._prepare_tool_as_response_schema(_response_schema_as_tool)
        _tools_dict = self._get_tools_dict(_tools)

        messages = self.load_history()

        if self.planning:
            messages.extend(
                await self._planning_step_async(
                    messages=messages,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    **all_kwargs,
                )
            )
        else:
            messages.extend(self._process_user_prompt(prompt))

        ctx = AgentInternalContext(
            max_error_retries=self.max_error_retries,
            max_repeat_tool_call=len(_tools) + 2,
        )
        final_result = None
        while ctx:
            streamed_response = self.llm.stream_text_async(
                messages=messages,
                system_instruction=system_instruction,
                tools=_tools,
                **all_kwargs,
            )
            async for chunk in streamed_response:
                if not chunk.is_last_chunk:
                    yield chunk

            messages.append(chunk)

            self._usage = self._update_usage(self._usage, chunk)

            message_next_turn, is_error, final_result = (
                self._validate_tools_and_prepare_next_message(
                    chunk, _tools_dict, _response_schema_as_tool
                )
            )

            if message_next_turn:
                messages.append(message_next_turn)

            if final_result:
                self.logger.debug(f"Final result: {final_result.contents}")
                messages.append(final_result)
                final_result.usage = self._usage
                yield final_result
                break
            else:
                if is_error:
                    ctx.increment_error_retries_count()
                    ctx.increment_repeat_tool_call_count()
                    messages.append(
                        Prompt(contents=[TextPrompt(text=self.retry_prompt)], role="user")
                    )
                else:
                    ctx.reset_error_retries_count()

                    if _response_schema_as_tool is None:
                        if self._include_tool_call_response(chunk):
                            ctx.increment_repeat_tool_call_count()
                            ctx.reset_repeat_text_response_count()
                            continue
                        else:
                            break
                    else:
                        if self._include_tool_call_response(chunk):
                            ctx.reset_repeat_text_response_count()
                            ctx.increment_repeat_tool_call_count()
                        else:
                            ctx.increment_repeat_text_response_count()
                            ctx.reset_repeat_tool_call_count()

                            if ctx.text_response_count != ctx.max_repeat_text_response:
                                messages.append(
                                    Prompt(contents=self.__no_tool_use_retry_prompt, role="user")
                                )

        if not ctx:
            raise RetryLimitExceededError()

        self.store_history(messages)

        chunk.usage = self._usage
        yield chunk

    def generate_image(self, prompt: str, **kwargs: Any) -> Response:
        """Generate image response from the model.

        Parameters
        ----------
        prompt : str
            The user input to the agent.
        **kwargs : Any
            Additional keyword arguments to pass to the API provider.
            Basically the same as the parameters in the original provider API.
            It means the agent accepts original parameters of OpenAI API if you use OpenAIClient in this agent.
            For more details, please refer to the API reference of each provider.

        Returns
        ----------
        Response
            The response from the model containing ImageResponse.
        """
        all_kwargs = {**self.init_kwargs, **kwargs}
        return self.llm.generate_image(prompt, **all_kwargs)

    async def generate_image_async(self, prompt: str, **kwargs: Any) -> Response:
        """Generate image response from the model asynchronously.

        Parameters
        ----------
        prompt : str
            The user input to the agent.
        **kwargs : Any
            Additional keyword arguments to pass to the API provider.
            Basically the same as the parameters in the original provider API.
            It means the agent accepts original parameters of OpenAI API if you use OpenAIClient in this agent.
            For more details, please refer to the API reference of each provider.

        Returns
        ----------
        Response
            The response from the model containing ImageResponse.
        """
        all_kwargs = {**self.init_kwargs, **kwargs}
        return await self.llm.generate_image_async(prompt, **all_kwargs)

    def generate_audio(
        self,
        prompt: AgentInput,
        **kwargs: Any,
    ) -> Response:
        """Generate audio response from the model.

        Parameters
        ----------
        prompt : AgentInput
            The user input to the agent.
        **kwargs : Any
            Additional keyword arguments to pass to the API provider.
            Basically the same as the parameters in the original provider API.
            It means the agent accepts original parameters of OpenAI API if you use OpenAIClient in this agent.
            For more details, please refer to the API reference of each provider.

        Returns
        ----------
        Response
            The response from the model containing AudioResponse.
        """
        all_kwargs = {**self.init_kwargs, **kwargs}
        messages = self.load_history()
        messages.extend(self._process_user_prompt(prompt))
        response = self.llm.generate_audio(messages, **all_kwargs)

        messages.append(response)
        self.store_history(messages)
        return response

    async def generate_audio_async(
        self,
        prompt: AgentInput,
        **kwargs: Any,
    ) -> Response:
        """Generate audio response from the model asynchronously.

        Parameters
        ----------
        prompt : AgentInput
            The user input to the agent.
        **kwargs : Any
            Additional keyword arguments to pass to the API provider.
            Basically the same as the parameters in the original provider API.
            It means the agent accepts original parameters of OpenAI API if you use OpenAIClient in this agent.
            For more details, please refer to the API reference of each provider.

        Returns
        ----------
        Response
            The response from the model containing AudioResponse.
        """
        all_kwargs = {**self.init_kwargs, **kwargs}
        messages = self.load_history()

        messages.extend(self._process_user_prompt(prompt))
        response = await self.llm.generate_audio_async(messages, **all_kwargs)

        messages.append(response)
        self.store_history(messages)
        return response

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        all_kwargs = {**self.init_kwargs, **kwargs}
        return self.llm.embed_text(texts, **all_kwargs)

    async def embed_text_async(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingResults:
        all_kwargs = {**self.init_kwargs, **kwargs}
        return await self.llm.embed_text_async(texts, **all_kwargs)

    def _process_user_prompt(self, prompt: AgentInput) -> list[Prompt | Response]:
        if isinstance(prompt, (Prompt, Response)):
            return [prompt]
        elif isinstance(prompt, (str, PromptType)):
            return [Prompt(role="user", contents=prompt)]
        elif isinstance(prompt, ResponseType):
            return [Response(role="assistant", contents=[prompt], usage=Usage())]
        elif isinstance(prompt, list):
            include_prompt_or_response = False
            include_content = False

            for p in prompt:
                if isinstance(p, (Prompt, Response)):
                    include_prompt_or_response = True
                elif isinstance(p, (str, PromptType)):
                    include_content = True

            if include_prompt_or_response and include_content:
                raise ValueError(
                    "Prompt types or roles are ambiguous. Don't mix Prompt/Response and str/content."
                )

            if include_prompt_or_response:
                messages = prompt
            elif include_content:
                messages = [Prompt(role="user", contents=prompt)]
            else:
                raise ValueError("Invalid prompt type")

            return messages
        else:
            raise ValueError("Invalid prompt type. Please provide the correct prompt type.")

    def _is_tool_call_response(self, content: ResponseType) -> bool:
        if isinstance(content, ToolCallResponse):
            return True
        return False

    def _is_text_response(self, content: ResponseType) -> bool:
        if isinstance(content, TextResponse):
            return True
        return False

    def _validate_tools_and_prepare_next_message(
        self,
        response: Response,
        tools_dict: dict[str, Tool],
        response_schema_as_tool: BaseModel | None,
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

                try:
                    args = json.loads(content.args or "{}")
                    self.logger.info(f"Running tool: {tool_name}")

                    if tool_name != self.__response_schema_name:
                        assert (
                            schema_validator is not None
                        ), f"Schema validator is required for tool: {tool_name} but it's NoneType object. "
                        args = schema_validator.validate_python({**args, **context})
                        tool_result = tool.run(args)
                        next_turn_contents.append(
                            ToolUsePrompt(
                                output=tool_result,
                                call_id=content.call_id,
                                args=content.args,
                                name=tool_name,
                            )
                        )

                    else:
                        assert (
                            response_schema_as_tool is not None
                        ), "Please provide response_schema_as_tool in the agent."

                        # validate response schema
                        response_schema_as_tool.model_validate(args)

                        next_turn_contents.append(
                            ToolUsePrompt(
                                output=json.dumps(args, ensure_ascii=False),
                                call_id=content.call_id,
                                args=content.args,
                                name=tool_name,
                            )
                        )

                        # Create a artificial Response in the assistant role
                        # to return a TextResponse as the final response.
                        # It also needs to continue the conversation because, for example,
                        # Claude requires the user role and the assistant role to enter alternating conversations
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
                    msg = (
                        f"Error occurred while running tool: {tool_name}. Retrying...\n"
                        f"Input value: {content.args}\n"
                        f"{e.__class__.__name__}: {str(e)}"
                    )
                    self.logger.error(msg)

                    next_turn_contents.append(
                        ToolUsePrompt(
                            error=msg,
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
                        include={"role", "contents", "name", "type"}, exclude={"raw", "usage"}
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
                description=self.agent_config.final_answer_description,
                schema_dict=response_schema_as_tool,
            )
        ]


def _generate_dynamic_tool_as_agent(
    agent: Agent,  # type: ignore
    agent_name: str,
) -> Tool:
    tool_name = f"route_{agent_name}"
    duplicated_tool = _duplicate_function(_run_subagent, tool_name)

    agent_capabilities = _get_agent_capabilities(agent)

    return Tool(
        tool=duplicated_tool,
        name=tool_name,
        description=(
            f"This function invokes the agent capable to run the following tools or agents:\n"
            f"{agent_capabilities}"
        ),
        context={"agent": agent, "agent_name": agent_name},
    )


def _get_agent_capabilities(
    subagent: Agent,  # type: ignore
    indent: int = -1,
) -> str:
    if indent == -1:
        txt = ""
    else:
        txt = "  " * (indent) + "- " + subagent._name + "\n"

    _subagents = subagent.subagents or []

    if _subagents:
        for _subagent in _subagents:
            if isinstance(_subagent, Agent):
                txt += _get_agent_capabilities(_subagent, indent + 1)
            else:
                raise ValueError(
                    "Subagent must be an instance of Agent class. "
                    "Please provide the correct agent instance."
                )

    indent_space = "  " * (indent + 1)
    # for tool in subagent._tools:
    tools = subagent.tools or []
    for tool in tools:
        if _tool := tool.tool:
            if signature := inspect.signature(_tool):
                if agent_signature := signature.parameters.get("agent"):
                    if agent_signature.annotation is not Agent:
                        txt += f"{indent_space}- {tool.name}: {tool.description}\n"
                else:
                    if tool.name != "final_answer":
                        txt += f"{indent_space}- {tool.name}: {tool.description}\n"

    return txt


def _run_subagent(
    agent: Agent,  # type: ignore
    agent_name: str,
    instruction: str,
) -> str:
    """
    This function is used to run the agent from the parent agent.

    Parameters
    ----------
    agent : Agent
        The agent instance.
    agent_name : str
        The name of the agent.
    instruction : str
        The detail and specific instruction to the agent, including the plan to get answer.

    Returns
    ----------
    str
        The response from the agent.
    """
    if not isinstance(agent, Agent):
        raise ValueError(
            "Subagent must be an instance of Agent class. "
            "Please provide the correct agent instance."
        )

    return agent.generate_text(instruction, name=agent_name).contents[0].text  # type: ignore


def _duplicate_function(func: Callable[..., Any], name: str) -> Callable[..., Any]:
    """
    This function dynamically duplicates the function with a new name.
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
