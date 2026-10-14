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

# Configuration constants
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

# Helper function to invoke LLM safely with retries for temporary spikes
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

# 1. DEFINE STATE WITH ITERATIONS COUNTER
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int

# 2. DEFINE AGENT NODE WITH ITERATION TRACKING
def agent_node(state: AgentState) -> dict:
    current_iterations = state.get("iterations", 0) + 1
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    messages = state["messages"]
    print(f"\n[NODE: agent] Iteration {current_iterations}/{MAX_ITERATIONS} | Messages: {len(messages)}")

    # If approaching max iterations, inject a prompt directive to wrap up
    if current_iterations >= MAX_ITERATIONS:
        print(f"[WARN] Reached maximum iterations ({MAX_ITERATIONS}). Forcing final response synthesis.")
        system_override = SystemMessage(content="IMPORTANT: You have reached the maximum allowed tool call limit. Do NOT call any more tools. Summarize your findings based on the existing conversation history.")
        response = invoke_with_retry(llm, messages + [system_override])
    else:
        response = invoke_with_retry(llm_with_tools, messages)
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = ", ".join([tc['name'] for tc in response.tool_calls])
        print(f"[NODE: agent] Requesting Tool Execution: {tool_names}")
    else:
        print("[NODE: agent] Final Answer synthesized.")

    return {
        "messages": [response],
        "iterations": current_iterations
    }

# 3. DEFINE CONDITIONAL ROUTING WITH MAX ITERATION SAFEGUARD
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    iterations = state.get("iterations", 0)

    # Safeguard 1: Hard iteration cap
    if iterations >= MAX_ITERATIONS:
        print(f"[ROUTER] 🛑 MAX_ITERATIONS ({MAX_ITERATIONS}) reached. Routing to END to prevent infinite loops.")
        return END

    # Safeguard 2: Check if tool calls exist
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(f"[ROUTER] Tool call requested -> Routing to 'tools' node (Iteration {iterations}).")
        return "tools"
    
    print("[ROUTER] No tool calls requested -> Routing to END.")
    return END

# 4. BUILD THE AGENT LOOP GRAPH
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    builder.add_edge(START, "agent")

    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    # Loop edge back to agent
    builder.add_edge("tools", "agent")

    return builder.compile()

def run_test(title: str, query: str):
    print("\n" + "="*75)
    print(f"TEST: {title}")
    print(f"QUERY: '{query}'")
    print("="*75)

    app = build_graph()
    initial_state = {
        "messages": [
            SystemMessage(content="You are an AI Decision Agent. Use web_search to find facts, calculator to compute stats, and text_analysis to evaluate docs."),
            HumanMessage(content=query)
        ],
        "iterations": 0
    }

    result = app.invoke(initial_state)

    print("\n[GRAPH FINISHED] Final Result:")
    final_msg = result["messages"][-1]
    content_text = final_msg.content
    if isinstance(content_text, list):
        content_text = "\n".join([item.get("text", "") for item in content_text if isinstance(item, dict) and "text" in item])
    print("-" * 65)
    print(content_text)
    print("-" * 65)

def main():
    print("[INFO] Building Phase 6 Agent Loop with Max Iteration Safeguard...")

    # Test 1: Multi-Tool Sequential Research + Calculation Loop
    run_test(
        "Multi-Tool Sequential Execution",
        "Search for the current pricing of NVIDIA H100 GPU and AMD MI300X GPU. Then calculate the cost difference between buying 8 NVIDIA H100s vs 8 AMD MI300Xs."
    )

if __name__ == "__main__":
    main()
