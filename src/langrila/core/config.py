from pydantic import Field

from .pydantic import BaseModel
from .retry import RetryPrompt


class AgentConfig(BaseModel):
    retry_prompt: RetryPrompt = Field(
        default=RetryPrompt(),
        description="Retry prompt configuration",
    )

    n_validation_retries: int = Field(
        default=3,
        description="Number of retries when an error occurs",
    )
