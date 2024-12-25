from pydantic import Field

from .pydantic import BaseModel


class InternalPrompt(BaseModel):
    error_retry: str = Field(
        default=("Please fix the error on tool use."),
        description="Retry prompt when validation error is raised",
    )

    no_tool_use_retry: str = Field(
        default=(
            "Briefly reflect the conversation history and plan the next action to take. "
            "You can select the next action from the following options.\n"
            "1. Invoke 'final_answer' tool based on the current state\n"
            "2. Invoke other tools with tentative specification to get additional information\n"
            "3. Invoke 'final_answer' tool forcibly and wait for the user's feedback\n"
        ),
        description="Prompt when no tool use is detected while `response_schema_as_tool' is specified.",
    )

    system_instruction: str = Field(
        default=(
            "You are an AI agent to support user. "
            "If the user want to speceify additional system instruction, you can find it bellow."
        ),
        description="Master system instruction that is automatically inserted into all the agent.",
    )
