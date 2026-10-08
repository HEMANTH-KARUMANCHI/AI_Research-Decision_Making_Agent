import os
import sys
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tools.calculator import calculator

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
        print("[INFO] Using ChatOpenAI (gpt-4o-mini)")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif google_key and not google_key.startswith("your_"):
        print("[INFO] Using ChatGoogleGenerativeAI (gemini-3.5-flash)")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=google_key, temperature=0)
    else:
        raise ValueError("No valid LLM key found in .env")

def main():
    print("[INFO] Initializing Phase 3 — Calculator Tool Integration...")
    llm = get_llm()

    # 1. BIND TOOL TO LLM
    # 'bind_tools' attaches JSON schemas of specified tools to the LLM
    llm_with_tools = llm.bind_tools([calculator])

    # 2. CREATE USER QUESTION REQUIRING MATH
    messages = [
        SystemMessage(content="You are a helpful decision-making assistant. Use tools when precision calculations are needed."),
        HumanMessage(content="If I invest ₹20,000 every month for 5 years, what is my total contribution?")
    ]

    print("\n[STEP 1] Sending prompt to LLM with calculator tool bound...")
    response = llm_with_tools.invoke(messages)

    # 3. INSPECT LLM RESPONSE FOR TOOL CALLS
    print(f"\n[STEP 2] LLM Response Type: {type(response).__name__}")
    print(f"Tool Calls Requested: {response.tool_calls}")

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"\n[STEP 3] Executing Tool Call: '{tool_call['name']}' with args: {tool_call['args']}")
        
        # Execute calculator tool directly
        tool_result = calculator.invoke(tool_call["args"])
        print(f"[TOOL RESULT] Calculator Output: {tool_result}")

        # Construct ToolMessage to send back to LLM
        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )

        # Append response and tool_message to conversation history
        messages.append(response)
        messages.append(tool_message)

        print("\n[STEP 4] Sending tool result back to LLM for final synthesis...")
        final_response = llm_with_tools.invoke(messages)

        print("\n[SUCCESS] Final Answer from Agent:")
        print("-" * 60)
        content_text = final_response.content
        if isinstance(content_text, list):
            content_text = "\n".join([item.get("text", "") for item in content_text if isinstance(item, dict) and "text" in item])
        print(content_text)
        print("-" * 60)
    else:
        print("[INFO] LLM did not request any tool calls. Direct response:")
        print(response.content)

if __name__ == "__main__":
    main()
