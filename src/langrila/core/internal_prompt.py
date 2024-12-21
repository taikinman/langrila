from pydantic import Field

from .pydantic import BaseModel


class InternalPrompt(BaseModel):
    validation_error_retry: str = Field(
        default="Please fix the error. If the fix is difficult, please provide a revision plan.",
        description="Retry prompt when validation error is raised",
    )

    review: str = Field(
        default=(
            "Based on the conversation, "
            "review if there is any missing information to answer, "
            "then decide to run tools or give final answer."
        ),
        description="Prompt for review before the final answer",
    )
