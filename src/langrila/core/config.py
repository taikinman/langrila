from pydantic import Field

from .internal_prompt import InternalPrompt
from .pydantic import BaseModel


class AgentConfig(BaseModel):
    internal_prompt: InternalPrompt = Field(
        default=InternalPrompt(),
        description="Retry prompt configuration",
    )

    n_validation_retries: int = Field(
        default=3,
        description="Number of retries when an error occurs",
    )

    store_conversation: bool = Field(
        default=True,
        description="Whether to store the conversation",
    )
