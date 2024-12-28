from dataclasses import dataclass


@dataclass
class AgentInternalContext:
    error_retries_count: int = 0
    max_error_retries: int = 3
    text_response_count: int = 0
    max_repeat_text_response: int = 3
    tool_call_count: int = 0
    max_repeat_tool_call: int = 3

    def __bool__(self) -> bool:
        return (
            self.error_retries_count < self.max_error_retries
            and self.text_response_count < self.max_repeat_text_response
            and self.tool_call_count < self.max_repeat_tool_call
        )

    def increment_error_retries_count(self) -> None:
        self.error_retries_count += 1

    def increment_repeat_text_response_count(self) -> None:
        self.text_response_count += 1

    def increment_repeat_tool_call_count(self) -> None:
        self.tool_call_count += 1

    def reset_error_retries_count(self) -> None:
        self.error_retries_count = 0

    def reset_repeat_text_response_count(self) -> None:
        self.text_response_count = 0

    def reset_repeat_tool_call_count(self) -> None:
        self.tool_call_count = 0
