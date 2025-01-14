class RetryLimitExceededError(Exception):
    def __init__(
        self,
        message: str = (
            "Retry limit exceeded. Please try again or check the tool configuration. "
            "Increasing the `max_error_retries` via the AgentConfig or "
            "more detail description of the tool and argument may help as well."
        ),
    ) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class ToolCallLimitExceededError(Exception):
    def __init__(
        self,
        message: str = (
            "Tool call limit exceeded. If the parallel tool calling does not work well, is not supported, "
            "or sequential tool calling is preferable, please try to increase the `max_consecutive_tool_call` "
            "via the AgentConfig as the agent may have to call the same tool multiple times. \n"
            "Otherwise, tools or subagents may not be well-configured."
        ),
    ) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class TextResponseLimitExceededError(Exception):
    def __init__(
        self,
        message: str = (
            "Text response limit exceeded. "
            "In case consecutive text response generation is the expected behaviour, please try to "
            "increase the `max_consecutive_text_response` via the AgentConfig. \n"
            "Otherwise, please check the tool configuration and the conversation history or log messages "
            "as this error is usually caused by the missing or incorrect configuration of the tool."
        ),
    ) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message
