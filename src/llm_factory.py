import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


class LLMFactory:
    """
    Factory class to create unified ChatModel instances for different providers.
    Supports: Google Gemini, Anthropic Claude, and OpenRouter (for DeepSeek/Llama).
    """

    @staticmethod
    def create_llm(
        provider: str,
        api_key: str,
        model_name: Optional[str] = None,
        temperature: float = 0.8
    ) -> BaseChatModel:
        """
        Args:
            provider: "Google", "Anthropic", or "OpenRouter"
            api_key: The API key for the selected provider.
            model_name: Specific model identifier (optional, defaults will be used).
            temperature: Creativity setting (0.0 to 1.0).
        """

        if provider == "Google":
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-2.5-pro",
                google_api_key=api_key,
                temperature=temperature,
                convert_system_message_to_human=True  # Fix for some Gemini versions
            )

        elif provider == "Anthropic":
            return ChatAnthropic(
                model=model_name or "claude-3-5-sonnet-20240620",
                api_key=api_key,
                temperature=temperature
            )

        elif provider == "OpenRouter":
            # OpenRouter uses the OpenAI SDK structure
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                model=model_name or "deepseek/deepseek-r1",
                temperature=temperature
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")
