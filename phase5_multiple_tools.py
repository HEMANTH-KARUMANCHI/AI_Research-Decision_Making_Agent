import os
import sys
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

# List of all available tools
ALL_TOOLS = [web_search, calculator, text_analysis]

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

# 1. DEFINE STATE
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. DEFINE AGENT NODE WITH ALL 3 TOOLS BOUND
def agent_node(state: AgentState) -> dict:
    llm = get_llm()
    # Bind all 3 tools: web_search, calculator, text_analysis
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    messages = state["messages"]
    print(f"\n[NODE: agent] Inspecting state history ({len(messages)} message(s))...")
    
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        tool_names = ", ".join([tc['name'] for tc in response.tool_calls])
        print(f"[NODE: agent] Selected Tool(s): {tool_names}")
    else:
        print("[NODE: agent] Final Answer generated (no tool calls).")

    return {"messages": [response]}

# 3. DEFINE CONDITIONAL ROUTING FUNCTION
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("[ROUTER] Tool call requested -> Routing to 'tools' node.")
        return "tools"
    
    print("[ROUTER] No tool calls -> Routing to END.")
    return END

# 4. BUILD THE GRAPH WITH MULTIPLE TOOLS
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

    builder.add_edge("tools", "agent")

    return builder.compile()

def run_test(title: str, query: str):
    print("\n" + "="*70)
    print(f"TEST: {title}")
    print(f"QUERY: '{query[:80]}...'")
    print("="*70)

    app = build_graph()
    initial_state = {
        "messages": [
            SystemMessage(content="You are an AI Research & Decision-Making Agent equipped with web_search, calculator, and text_analysis tools."),
            HumanMessage(content=query)
        ]
    }

    result = app.invoke(initial_state)

    print("\n[GRAPH FINISHED] Output:")
    final_msg = result["messages"][-1]
    content_text = final_msg.content
    if isinstance(content_text, list):
        content_text = "\n".join([item.get("text", "") for item in content_text if isinstance(item, dict) and "text" in item])
    print("-" * 60)
    print(content_text)
    print("-" * 60)

def main():
    print("[INFO] Building Phase 5 Multi-Tool Graph...")

    # Test 1: Web Research Tool
    run_test(
        "Web Research Use Case",
        "What are the major differences between PostgreSQL and MongoDB?"
    )

    # Test 2: Calculator Tool
    run_test(
        "Calculation Use Case",
        "If an engineer works 40 hours per week at $85 per hour for 50 weeks, what is their annual income?"
    )

    # Test 3: Text Analysis Tool
    sample_text = (
        "We are looking for a Senior AI Engineer skilled in Python, SQL, Docker, Kubernetes, "
        "LangChain, LangGraph, Machine Learning, PyTorch, and cloud architectures on AWS or GCP."
    )
    run_test(
        "Text Analysis Use Case",
        f"Analyze the following text and give me key statistics: '{sample_text}'"
    )

if __name__ == "__main__":
    main()
