import os
import sys
import time
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

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
        raise ValueError("No valid LLM key found in .env (expected OPENAI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY)")

# 1. DEFINE STATE
# State is the data structure that passes between nodes in the graph.
# 'add_messages' is a reducer function that appends new messages to the existing list instead of overwriting.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. DEFINE NODE FUNCTION
def call_model(state: AgentState) -> dict:
    llm = get_llm()
    messages = state["messages"]
    
    print(f"[NODE: call_model] Processing {len(messages)} message(s)...")
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# 3. BUILD THE GRAPH
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("call_model", call_model)

    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)

    graph = builder.compile()
    return graph

def main():
    print("[INFO] Building Phase 2 Minimal LangGraph...")
    app = build_graph()

    print("[INFO] Invoking graph with initial human query...")
    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful AI assistant built with LangGraph."),
            HumanMessage(content="What is LangGraph and why is state important in a graph?")
        ]
    }

    result = app.invoke(initial_state)

    print("\n[SUCCESS] Graph Execution Complete!")
    print("-" * 60)
    final_message = result["messages"][-1]
    
    content_text = final_message.content
    if isinstance(content_text, list):
        content_text = "\n".join([item.get("text", "") for item in content_text if isinstance(item, dict) and "text" in item])
        
    print(f"Final Response ({type(final_message).__name__}):")
    print(content_text)
    print("-" * 60)

if __name__ == "__main__":
    main()
