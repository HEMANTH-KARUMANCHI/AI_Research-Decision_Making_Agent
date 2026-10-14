from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import END
from config.settings import MAX_ITERATIONS
from agent.state import AgentState

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Conditional edge router: routes to 'tools' if tool calls are requested and iterations < MAX_ITERATIONS, else END."""
    messages = state["messages"]
    last_message = messages[-1]
    iterations = state.get("iterations", 0)

    # Safeguard: Hard iteration limit
    if iterations >= MAX_ITERATIONS:
        return END

    # Route to tools if tool calls exist
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END
