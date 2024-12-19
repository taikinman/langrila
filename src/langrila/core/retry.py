from pydantic import Field

from .pydantic import BaseModel


class RetryPrompt(BaseModel):
    validation: str = Field(
        default="Please fix the error. If the fix is difficult, please provide a revision plan.",
        description="Retry prompt when validation error is raised",
    )
