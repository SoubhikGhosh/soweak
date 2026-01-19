"""
soweak Integration Examples

This package contains integration examples for popular LLM frameworks:
- LangChain
- OpenAI
- Google Generative AI / ADK

Installation:
    pip install soweak[langchain]  # For LangChain
    pip install soweak[openai]     # For OpenAI
    pip install soweak[google]     # For Google AI
    pip install soweak[all]        # All integrations

Usage:
    from examples.langchain_integration import SecureLangChainPipeline
    from examples.openai_integration import SecureOpenAIClient
    from examples.google_adk_integration import SecureGeminiClient
"""

__all__ = [
    "langchain_integration",
    "openai_integration",
    "google_adk_integration",
]
