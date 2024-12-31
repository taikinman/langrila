from pydantic import Field

from .pydantic import BaseModel


class InternalPrompt(BaseModel):
    error_retry: str = Field(
        default=(
            "Please fix the error on tool use. "
            "If the validation error is raised, reflect the conversation history and try to find the correct answer. "
            "If there is no fact to answer the user, try to run other tools to get necessary information. "
        ),
        description="Retry prompt when error is raised",
    )

    no_tool_use_retry: str = Field(
        default=(
            "Decide the next action based on the conversation. "
            "If you have all information for answering the user, run 'final_result' tool. "
            "If you need more information, invoke other tool to get necessary information. "
        ),
        description="Prompt when no tool use is detected while `response_schema_as_tool' is specified.",
    )

    planning: str = Field(
        default=(
            "Please make a concise plan to answer the following question/requirement, considering the conversation history.\n"
            "You can invoke the sub-agents or tools to answer the questions/requirements shown in the capabilities section.\n"
            "Agent has no description while the tools have a description.\n\n"
            "Question/Requirement:\n"
            "{user_input}\n\n"
            "Capabilities:\n"
            "{capabilities}"
        ),
        description=(
            "Prompt for planning how to answer the user's question when the agent is planning mode."
            "The planning prompt includes the signature of the user input and the capabilities of the agent."
            "The capabilities are the descriptions of the tools and subagent."
        ),
    )

    do_plan: str = Field(
        default="Put the plan into action.",
        description="Prompt for executing the plan when the agent is planning mode.",
    )
