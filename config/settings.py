import os
from dotenv import load_dotenv

load_dotenv()

# Maximum allowed agent reasoning loop iterations
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Model configuration
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

def get_llm():
    """Factory function returning the configured LLM client."""
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=DEFAULT_OPENAI_MODEL, temperature=0)
    elif GOOGLE_API_KEY and not GOOGLE_API_KEY.startswith("your_"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=DEFAULT_GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
    else:
        raise ValueError("No valid LLM key found in .env (expected OPENAI_API_KEY or GEMINI_API_KEY)")
