from pydantic import Field

from .internal_prompt import InternalPrompt
from .pydantic import BaseModel


class AgentConfig(BaseModel):
    """
    The instance of this class is used to configure the internal behaviour and internal prompts in an agent
    and can be passed to the Agent constructor.
    """

    internal_prompt: InternalPrompt = Field(
        default=InternalPrompt(),
        description="Retry prompt configuration",
    )

    max_error_retries: int = Field(
        default=3,
        description="Number of retries when an error occurs",
    )

    store_conversation: bool = Field(
        default=True,
        description="Whether to store the conversation",
    )

    final_answer_description: str = Field(
        default=(
            "The final answer which ends this conversation. "
            "Arguments of this tool must be selected from the conversation history.\n"
            "Unkown argument in the entire conversation history must be null, "
            "however, the argument appeared in the previous conversation must be provided.\n"
        ),
        description=(
            "Description of the tool for generating final answer with specified response schema. "
            "This tool is invoked when 'response_schema_as_tool' is specified."
        ),
    )

    max_consecutive_tool_call: int | None = Field(
        default=None,
        description="Maximum number of times tools can be called in a row. If None, `len(agent.tools) + 2` is used.",
    )

    max_consecutive_text_response: int = Field(
        default=3,
        description="Maximum number of times text responses can be generated in a row. Default is 3.",
    )
