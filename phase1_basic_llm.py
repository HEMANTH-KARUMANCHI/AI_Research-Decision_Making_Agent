import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()

def get_llm():
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if openai_key and not openai_key.startswith("your_"):
        print("[INFO] Initializing OpenAI Chat Model (gpt-4o-mini)...")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif google_key and not google_key.startswith("your_"):
        print("[INFO] Initializing Google Gemini Chat Model (gemini-3.8-flash)...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=google_key, temperature=0)
    else:
        print("[ERROR] Neither OPENAI_API_KEY, GOOGLE_API_KEY, nor GEMINI_API_KEY found in .env file.")
        return None

def main():
    llm = get_llm()
    if not llm:
        return

    messages = [
        SystemMessage(content="You are a helpful AI research assistant."),
        HumanMessage(content="Explain the difference between a traditional script and an Agentic AI in 2 sentences.")
    ]

    print("[INFO] Sending request to LLM...")
    try:
        response = llm.invoke(messages)
        print("\n[SUCCESS] LLM Response:")
        print("-" * 50)
        content_text = response.content
        if isinstance(content_text, list):
            content_text = "\n".join([item.get("text", "") for item in content_text if isinstance(item, dict) and "text" in item])
        print(content_text)
        print("-" * 50)
    except Exception as e:
        print(f"[ERROR] LLM Invocation failed: {e}")

if __name__ == "__main__":
    main()