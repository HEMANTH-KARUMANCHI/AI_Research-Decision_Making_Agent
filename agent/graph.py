from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.state import AgentState
from agent.nodes import agent_node, ALL_TOOLS
from agent.router import should_continue

def build_graph():
    """Constructs and compiles the full LangGraph workflow."""
    builder = StateGraph(AgentState)

    # 1. Add Nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    # 2. Add Edges
    builder.add_edge(START, "agent")

    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    # Loop back from tools node to agent node
    builder.add_edge("tools", "agent")

    return builder.compile()
