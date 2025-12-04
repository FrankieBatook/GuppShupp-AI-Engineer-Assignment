from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

# --- Data Models ---


class MemorySchema(BaseModel):
    """Schema for structured extracted memory."""
    user_preferences: List[str] = Field(
        description="Specific likes, dislikes, habits, or hobbies mentioned by the user."
    )
    emotional_patterns: List[str] = Field(
        description="Recurring emotional states, triggers, or distinct mood shifts."
    )
    important_facts: List[str] = Field(
        description="Concrete details worth remembering (e.g., names, deadlines, goals, events)."
    )


class MemoryAgent:
    """
    Agent responsible for analyzing chat logs and extracting structured insights.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=MemorySchema)

    def extract_from_history(self, messages: List[str]) -> dict:
        """
        Analyzes a list of raw chat strings and returns a structured profile.
        """

        # System prompt designed for analysis, not conversation
        system_prompt = (
            "You are a backend memory processor for an AI companion. "
            "Your goal is to read the provided chat logs and extract structured metadata "
            "about the user. Do not respond to the messages; only analyze them."
            "\n\n{format_instructions}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Chat Logs:\n{chat_logs}")
        ])

        # Chain: Prompt -> LLM -> JSON Parser
        chain = prompt | self.llm | self.parser

        # Convert list to single string block
        logs_text = "\n".join([f"- {msg}" for msg in messages])

        try:
            return chain.invoke({
                "chat_logs": logs_text,
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            # Fallback for parsing errors
            return {"error": f"Failed to parse memory: {str(e)}"}
