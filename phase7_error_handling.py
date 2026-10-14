import os
import sys
import time
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import calculator, web_search, text_analysis

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()

MAX_ITERATIONS = 5
ALL_TOOLS = [web_search, calculator, text_analysis]

def get_llm():
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if openai_key and not openai_key.startswith("your_"):
        print("[INFO] Using ChatOpenAI (gpt-4o-mini)")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif google_key and not google_key.startswith("your_"):
        print("[INFO] Using ChatGoogleGenerativeAI (gemini-3.5-flash-lite)")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=google_key, temperature=0)
    else:
        raise ValueError("No valid LLM key found in .env")

# Helper with retry and error capture
def invoke_with_retry(llm_runnable, messages, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            return llm_runnable.invoke(messages)
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries:
                wait_time = attempt * 2
                print(f"[WARN] 503 Server Busy. Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e

# 1. DEFINE STATE
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int

# 2. DEFINE AGENT NODE WITH ERROR AWARENESS
def agent_node(state: AgentState) -> dict:
    current_iterations = state.get("iterations", 0) + 1
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    messages = state["messages"]
    print(f"\n[NODE: agent] Iteration {current_iterations}/{MAX_ITERATIONS} | Message Count: {len(messages)}")

    system_guidelines = (
        "You are an AI Research & Decision-Making Agent.\n"
        "Guidelines:\n"
        "1. If a tool returns an error, acknowledge the error and explain the limitation to the user.\n"
        "2. If a user asks for actions you have no tools for (e.g., booking tickets, deleting files), state clearly that you cannot perform that action.\n"
        "3. Never hallucinate having executed an unsupported action."
    )

    if current_iterations >= MAX_ITERATIONS:
        print(f"[WARN] Reached maximum iterations ({MAX_ITERATIONS}). Forcing final response.")
        override_msg = SystemMessage(content="You have reached the maximum tool iteration limit. Summarize your findings based on available history.")
        response = invoke_with_retry(llm, messages + [override_msg])
    else:
        response = invoke_with_retry(llm_with_tools, messages + [SystemMessage(content=system_guidelines)])
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = ", ".join([tc['name'] for tc in response.tool_calls])
        print(f"[NODE: agent] Requested Tool: {tool_names}")
    else:
        print("[NODE: agent] Final Answer formulated.")

    return {
        "messages": [response],
        "iterations": current_iterations
    }

# 3. ROUTER
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    iterations = state.get("iterations", 0)

    if iterations >= MAX_ITERATIONS:
        print(f"[ROUTER] 🛑 MAX_ITERATIONS reached -> Routing to END.")
        return END

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(f"[ROUTER] Routing to 'tools' node.")
        return "tools"
    
    print("[ROUTER] No tool calls -> Routing to END.")
    return END

# 4. BUILD GRAPH
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile()

def test_case(name: str, query: str):
    print("\n" + "="*75)
    print(f"TEST CASE: {name}")
    print(f"INPUT: '{query}'")
    print("="*75)

    app = build_graph()
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "iterations": 0
    }

    result = app.invoke(initial_state)

    print("\n[RESULT]")
    final_msg = result["messages"][-1]
    content_text = final_msg.content
    if isinstance(content_text, list):
        content_text = "\n".join([item.get("text", "") for item in content_text if isinstance(item, dict) and "text" in item])
    print("-" * 65)
    print(content_text)
    print("-" * 65)

def main():
    print("[INFO] Building Phase 7 Error Handling & Edge Cases Test Suite...")

    # Case 1: Invalid Math Expression (e.g. division by zero / malformed math)
    test_case(
        "Invalid Tool Input (Math Division by Zero)",
        "Please calculate 100 / 0 using the calculator tool."
    )

    # Case 2: Unsupported User Request (No tool available)
    test_case(
        "Unsupported Capability (Flight Booking)",
        "Book me a flight ticket from New York to London for next Monday."
    )

if __name__ == "__main__":
    main()
